import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import json
import time
from gym_env.enums import Action, Stage
import logging
import os

nb_max_start_steps = 1  # Не используется (заглушка)
nb_steps = 100000

logger = logging.getLogger(__name__)

# Define the Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN Agent in PyTorch
class Player:
    def __init__(
            self,
            name='DQN',
            load_model=None,
            env=None,
            gamma=0.995,  # Уменьшен для более быстрого получения наград
            lr=0.0003,  # Оптимизированная скорость обучения
            epsilon=0.5,  # Более сбалансированное исследование
            epsilon_min=0.05,  # Минимальный уровень исследования
            epsilon_decay=0.9995,  # Медленное затухание исследования
            memory_limit=100000,  # Увеличенный буфер памяти
            batch_size=512  # Оптимальный размер батча
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_limit)
        self.state_size = env.observation_space[0]
        self.action_size = env.action_space.n
        self.last_action = None

        # Define policy and target networks
        self.policy_net = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to inference mode

        # Load model if specified
        if load_model:
            self.load(f'models/dqn_{load_model}_weights.pth')
            # Загрузить метаданные
            with open(f'models/dqn_{load_model}_meta.json', 'r') as f:
                metadata = json.load(f)
                self.epsilon = metadata['epsilon']
                self.start_episode = metadata['episode'] + 1  # Следующий эпизод
                self.total_steps = metadata['total_steps']
        else:
            self.start_episode = 0
            self.total_steps = 0

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            weight_decay=1e-4  # L2 регуляризация
        )

    def _cleanup_old_checkpoints(self, env_name, keep_last=3):
        """Удаляет старые чекпоинты, оставляя только последние `keep_last`"""
        checkpoints = sorted(
            [f for f in os.listdir("models") if f.startswith(f"dqn_{env_name}")],
            key=lambda x: os.path.getmtime(os.path.join("models", x)),
            reverse=True
        )

        for old_checkpoint in checkpoints[keep_last:]:
            os.remove(os.path.join("models", old_checkpoint))
            if os.path.exists(old_checkpoint.replace(".pth", "_meta.json")):
                os.remove(old_checkpoint.replace(".pth", "_meta.json"))

    def train(self, env_name, model_name=None, nb_max_start_steps=1, start_step_policy=None, log_dir='./Graph'):
        """Train a model"""
        # Initialize TensorBoard
        logger.info("Training DQN")
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        writer = SummaryWriter(log_dir=f'{log_dir}/{timestr}')

        total_steps = 0
        for episode in range(self.start_episode, self.start_episode + 1500):
            print(f"Episode {episode}")
            state = self.env.reset()
            self.last_action = None
            done = False
            episode_reward = 0
            steps = 0
            episode_loss = []

            while not done:
                # if steps < nb_max_start_steps and start_step_policy:
                #     action = start_step_policy()  # Custom initial action policy if specified
                #     print("ACTION TYPE", type(action))
                # else:
                state = np.nan_to_num(state, nan=0.0)
                action = self.action(self.env.legal_moves, state, None)
                # print(action)
                # logger.info(f"Action: {action}")
                # if state == Stage.SHOWDOWN.value or state == Stage.SHOWDOWN:
                #     quit()
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.nan_to_num(next_state, nan=0.0)

                self.remember(state, action, reward, next_state, done)
                self.last_action = action
                state = next_state

                loss = self.replay()  # Train the model with experience replay

                # update the running average of the loss
                if loss:
                    episode_loss.append(loss)

                episode_reward += reward
                steps += 1
                total_steps += 1

            # Log the reward for this episode
            writer.add_scalar('Episode Reward', episode_reward, episode)
            writer.add_scalar('Steps per Episode', steps, episode)
            if episode_loss:
                episode_loss = np.array(episode_loss)
                writer.add_scalar('Loss/Actor Loss', np.mean(episode_loss), episode)

            # Update target network periodically
            if episode % 10 == 0:
                try:
                    # Обновить целевую сеть
                    self.update_target_network()

                    # Сохранить модель с уникальным именем
                    checkpoint_path = f"models/dqn_{env_name}_ep{episode}_weights.pth"
                    self.save(checkpoint_path)

                    # Дополнительно: сохранить метаданные
                    metadata = {
                        "episode": episode,
                        "epsilon": self.epsilon,
                        "total_steps": total_steps
                    }
                    with open(f"models/dqn_{env_name}_ep{episode}_meta.json", "w") as f:
                        json.dump(metadata, f)

                    # Удалить старые чекпоинты (оставляя последние 3)
                    self._cleanup_old_checkpoints(env_name, keep_last=3)

                except Exception as e:
                    logger.error(f"Checkpoint failed: {str(e)}")

            # Log progress every 100 episodes
            if episode % 100 == 0:
                print(f"Episode {episode}/{nb_steps} - Reward: {episode_reward:.2f} - Epsilon: {self.epsilon:.2f}")

        # Save the architecture and weights
        dqn_json = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "layers": [64, 64]  # Match the architecture details
        }
        with open(f"models/dqn_{env_name}_json.json", "w") as json_file:
            json.dump(dqn_json, json_file)

        self.save(f'models/dqn_{env_name}_weights.pth')

        # Test the model
        self.test(nb_episodes=5)

        writer.close()

    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        logger.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        return action
    def test(self, nb_episodes=5):
        """Evaluate the agent"""
        for episode in range(nb_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # action = self.act(state)  # Use the policy for testing (exploit only)
                state = np.nan_to_num(state, nan=0.0)
                action = self.action(self.env.legal_moves, state, None)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            print(f"Test Episode {episode+1}/{nb_episodes} - Reward: {episode_reward:.2f}")

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action based on epsilon-greedy policy, considering only legal moves."""
        # Check if we are exploring or exploiting
        if np.random.rand() <= self.epsilon:
            # Explore by choosing a random valid action
            return random.choice(self.env.legal_moves)

        # Get Q-values from the policy network
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze()
            q_values = q_values.detach().cpu().numpy()

        # Mask invalid actions by setting their Q-values to a very low number
        masked_q_values = np.full_like(q_values, -np.inf)
        for action in self.env.legal_moves:
            masked_q_values[action.value] = q_values[action.value]

        # Choose the action with the highest Q-value among legal moves
        return np.argmax(masked_q_values)


    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info
        # print(observation.shape)
        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        allowed_actions = list(this_player_action_space.intersection(set(action_space)))
        if Stage.SHOWDOWN in allowed_actions:
            allowed_actions.remove(Stage.SHOWDOWN)
        if Stage.SHOWDOWN.value in allowed_actions:
            allowed_actions.remove(Stage.SHOWDOWN.value)
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        # Get Q-values from the policy network for the current state
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze()
            q_values = q_values.detach().cpu().numpy()

        # Filter Q-values to only include valid actions
        # allowed_actions = list(action_space)
        # masked_q_values = {action: q_values[action.value] for action in allowed_actions}
        mask = np.full_like(q_values, -np.inf)
        for action in allowed_actions:
            mask[action.value] = q_values[action.value]
        # Select the action with the highest Q-value among allowed actions
        action = np.argmax(mask)

        #action = random.choice(list(possible_moves))
        print("\n--- Debug Info ---")
        print("Q-values:", q_values)
        print("Mask:", mask)
        print("Allowed actions:", [Action(a).name for a in allowed_actions])
        return action

    def replay(self):
        """Train the agent with a batch of experiences"""

        if len(self.memory) < self.batch_size:
            return
        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        # print(type(actions), actions)
        # actions = torch.LongTensor(actions).unsqueeze(1)
        actions = torch.LongTensor([action.value if type(action) != np.int64 else action for action in actions ]).unsqueeze(1).to(self.device)
        # print('actions converted to tensor')
        # logger.info(f'actions: {actions}')
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        print("\n--- Device Check ---")
        print(f"Policy Net: {next(self.policy_net.parameters()).device}")
        print(f"States: {states.device}")
        print(f"Next States: {next_states.device}")
        print(f"Actions: {actions.device}")
        # Q values for current states (policy net)
        current_q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Q values for next states (target net)
        next_q_values = self.target_net(next_states).max(1)[0]
        next_q_values[dones == 1] = 0.0  # Set next q values to 0 where episode ended

        # Compute target
        target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_values, target_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.cpu().item()

    def update_target_network(self):
        # Полная синхронизация весов целевой сети с политической
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self, filepath):
        """Load the model weights"""
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save the model weights"""
        torch.save(self.policy_net.state_dict(), filepath)

    def play(self, nb_episodes=5, render=False):
        """Играть с обученной моделью (без обучения)"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # Отключаем exploration
        self.test(nb_episodes=nb_episodes)
        self.epsilon = original_epsilon
