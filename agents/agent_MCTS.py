import numpy as np
import random
import math
from collections import defaultdict, deque
from gym_env.enums import Action
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MCTSNet(nn.Module):
    def __init__(self, input_size, num_actions):
        super(MCTSNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        
        # Recurrent layer for capturing sequential dependencies
        self.lstm = nn.LSTM(128, 128, batch_first=True, dropout=0.3)
        
        # Attention mechanism to allow the model to focus on key features
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = x[:, :]

        # Apply attention
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)

        return policy, value

class MinMaxStats:
    def __init__(self):
        self.min_value = float('inf')
        self.max_value = float('-inf')
    
    def update(self, value):
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value
    
    def normalize(self, value):
        if self.max_value == self.min_value:
            return 0
        return (value - self.min_value) / (self.max_value - self.min_value)
    
    def reset(self):
        self.min_value = float('inf')
        self.max_value = float('-inf')

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0  # Average reward value
        self.prior = prior  # For UCB and exploration
        self.reward = 0.0  # Reward value from rollouts or simulations
        self.done = False  # Whether the state is terminal

class MonteCarloTreeSearch:
    def __init__(self, env, neural_net, replay_buffer, max_depth=2000, num_simulations=10000, exploration_constant=1.0, dirichlet_alpha=0.03, exploration_fraction=0.25):
        self.env = env
        self.neural_net = neural_net
        self.replay_buffer = replay_buffer
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.min_max_stats = MinMaxStats()
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.optimizer = optim.Adam(neural_net.parameters(), lr=0.001)

    def search(self, state, action_space):
        root = MCTSNode(state)
        self.add_exploration_noise(root)
        for _ in range(self.num_simulations):
            self.simulate(root, action_space)
        action = self.best_action(root)
        buffer, next_state, reward, done, _ = self.env.mcts_step(action)
        print("ADDING TO REPLAY BUFFER")
        return self.best_action(root)

    def add_exploration_noise(self, root):
        actions = list(root.children)
        if actions:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            frac = self.exploration_fraction
            for action, n in zip(actions, noise):
                root.Q[action] = root.Q[action] * (1 - frac) + n * frac
                root.N[action] = 0

    def simulate(self, node, action_space, depth=0):
        if depth >= self.max_depth or node.done:
            return self.evaluate(node.state)

        if not node.children:
            return self.expand(node, action_space)

        child = self.select_child(node)
        value = self.simulate(child, action_space, depth + 1)
        self.backpropagate(child, value)
        return value
        

    def expand(self, node, action_space):
        action_space_list = list(action_space)
        if node.done:
            return 0  # End expansion for terminal states.

        state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.neural_net(state_tensor)

        action_probabilities = policy.squeeze().numpy()
        action_probabilities = action_probabilities[:len(action_space_list)]
        action_probabilities /= np.sum(action_probabilities)
        if np.any(np.isnan(action_probabilities)) or np.any(action_probabilities < 0):
            print("NaN or negative value detected in action probabilities")
            action_probabilities = np.ones(len(action_space_list))/len(action_space_list)
            
        action = np.random.choice(action_space_list, p=action_probabilities)
        print("*"*100)
        print("*"*100)
        print("HOLD UP WE EXPANDING")
        print("*"*100)
        print("*"*100)
        buffer, next_state, reward, done, _ = self.env.mcts_step(action)

        child = MCTSNode(next_state, parent=node, action=action)
        child.reward = reward
        child.done = done
        node.children.append(child)

        self.N[tuple(next_state)] = 0
        self.Q[tuple(next_state)] = 0.0

        if done:
            return reward  # Return immediate reward for terminal states

        rollout_value = self.rollout(next_state)
        return rollout_value

    def rollout(self, state):
        total_reward = 0
        done = False
        discount = 0.995
        while not done:
            action_space = self.env.legal_moves
            if not action_space:
                break
            action = random.choice(list(action_space))
            buffer, state, reward, done, _ = self.env.mcts_step(action)
            total_reward += reward * discount
        return total_reward

    def select_child(self, node):
        return max(node.children, key=lambda c: self.uct_value(c))

    def uct_value(self, node):
        parent_state = node.parent.state
        action = node.action
        q_value = self.Q[(tuple(parent_state), action)]
        visit_count = self.N[(tuple(parent_state), action)]
        parent_visits = node.parent.visits if node.parent else 1
        pb_c = math.log((parent_visits + 1) / 1) + 1.25
        pb_c *= math.sqrt(parent_visits) / (visit_count + 1e-10)

        prior_score = pb_c * node.prior

        if visit_count > 0:
            value_score = self.min_max_stats.normalize(
                node.reward + 0.995 * node.value
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, node, value):
        while node:
            parent_state = node.parent.state if node.parent else None
            action = node.action

            if parent_state is not None:
                self.N[(tuple(parent_state), action)] += 1
                current_q_value = self.Q[(tuple(parent_state), action)]
                visits = self.N[(tuple(parent_state), action)]
                self.Q[(tuple(parent_state), action)] += (value - current_q_value) / (visits + 1)

            node.visits += 1
            node.value += value
            node = node.parent

    def best_action(self, node):
        best_child = max(node.children, key=lambda c: c.visits)
        return best_child.action

    def evaluate(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value = self.neural_net(state_tensor)
        return value.item()
        
def train_neural_network(network, replay_buffer, optimizer, batch_size=32, gamma=0.995):
    if len(replay_buffer) < batch_size:
        print("Not enough data to train on")
        return
    
    # Sample a batch from the replay buffer
    weights = [1 + i for i in range(len(replay_buffer))]  # Give more weight to recent experiences
    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= weights.sum()  # Normalize to get probabilities
    indices = torch.multinomial(weights, batch_size, replacement=True)
    
    # Unpack the buffer and filter out None values in actions
    batch = [replay_buffer[i] for i in indices]
    filtered_batch = [(state, action, reward, done, info) for state, action, reward, done, info in batch if action is not None]
    
    # If after filtering, we have no valid data, return early
    if len(filtered_batch) == 0:
        print("No valid actions in the batch after filtering None values")
        return
    
    # Unpack the filtered batch into separate variables
    states, actions, rewards, dones, infos = zip(*filtered_batch)  # Unpack the filtered batch
    
    # Convert actions to their integer values
    action_values = [action.value for action in actions]  # Convert Action enum to its integer value
    actions_tensor = torch.tensor(action_values, dtype=torch.long)  # Create a tensor for actions
    
    # Convert other elements to tensors
    states = torch.tensor(states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)  # done flag as tensor (1.0 or 0.0)
    
    # Forward pass
    policies, values = network(states)
    
    # Compute target values
    # If done, use reward as target, otherwise use the value of the current state
    targets = rewards + gamma * (1 - dones) * values.squeeze()  # If done, don't propagate future rewards
    
    # Compute the policy loss (actor loss)
    policy_loss = -torch.mean(torch.log(policies.gather(1, actions_tensor.unsqueeze(1))) * (targets - values.squeeze()))
    
    # Compute the value loss (critic loss)
    value_loss = nn.MSELoss()(values.squeeze(), targets)
    
    # Combine losses
    loss = policy_loss + value_loss
    
    weighted_loss = (loss * weights[indices].unsqueeze(1)).mean()  # Weighted loss
    
    # Backpropagation
    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()


class Player:
    def __init__(self, env, neural_net, replay_buffer, name='MCTS'):
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.autoplay = True
        self.name = name
        self.env = env
        self.mcts = MonteCarloTreeSearch(env, neural_net, replay_buffer)

    def action(self, action_space, observation, info):
        # print("MCTS IS GOING")
        # print("-"*200)
        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT,
                                    Action.RAISE_HALF_POT, Action.RAISE_2POT}
        possible_moves = this_player_action_space.intersection(set(action_space))
        if not possible_moves:
            raise ValueError("No valid moves available")

        action = self.mcts.search(observation, possible_moves)
        return action