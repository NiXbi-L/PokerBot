import torch
import numpy as np
from enum import Enum
from torch import nn, optim
from collections import deque
from gym import Env
from gym.spaces import Discrete, Box
from gym_env.enums import Action, Stage


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


class DQNAgent:
    def __init__(self, state_size, action_size=8, load_model=None):
        self.state_size = state_size
        self.action_size = action_size

        # Define policy and target networks
        self.policy_net = DQNetwork(self.state_size, self.action_size)
        self.target_net = DQNetwork(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to inference mode

        if load_model:
            self.load(load_model)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def action(self, action_space, observation, info=None):
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
        state = torch.FloatTensor(observation).unsqueeze(0)

        # Get Q-values from the policy network for the current state
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze().numpy()

        # Filter Q-values to only include valid actions
        # allowed_actions = list(action_space)
        # masked_q_values = {action: q_values[action.value] for action in allowed_actions}
        mask = np.full_like(q_values, -np.inf)
        for action in allowed_actions:
            mask[action.value] = q_values[action.value]
        # Select the action with the highest Q-value among allowed actions
        action = np.argmax(mask)

        # action = random.choice(list(possible_moves))
        # print("\n--- Debug Info ---")
        # print("Q-values:", q_values)
        # print("Mask:", mask)
        # print("Allowed actions:", [Action(a).name for a in allowed_actions])
        return action


def create_observation_space(data: dict, big_blind: float) -> np.ndarray:
    """
    Формирует вектор observation_space из структурированных входных данных
    с фиксированным размером 328
    """
    # Нормализующий множитель
    norm_factor = big_blind * 100

    # 1. Извлекаем и нормализуем Community Data
    community = data['community']
    community_arr = [
        community['pot'] / norm_factor,
        community['round_pot'] / norm_factor,
        community['small_blind'],
        community['big_blind'],
        *community['stage_one_hot'],  # one-hot: [preflop, flop, turn, river]
        *community['legal_moves']  # бинарный вектор доступных действий
    ]

    # 2. Извлекаем и нормализуем Player Data
    player = data['player']
    player_arr = [
        player['position'],
        player['equity_alive'],
        player['equity_2plr'],
        player['equity_3plr'],
        player['stack'] / norm_factor
    ]

    # 3. Обрабатываем Stage Data для всех этапов
    stage_arr = []
    for stage in data['stages']:
        stage_features = [
            stage['calls'],
            stage['raises'],
            stage['min_call'] / norm_factor,
            stage['contribution'] / norm_factor,
            stage['stack_at_action'] / norm_factor,
            stage['community_pot_at_action'] / norm_factor
        ]
        stage_arr.extend(stage_features)

    # Объединяем все массивы
    observation = np.concatenate([
        np.array(player_arr),
        np.array(community_arr),
        np.array(stage_arr)
    ]).flatten()

    # Замена NaN и приведение к фиксированному размеру
    observation = np.nan_to_num(observation, nan=0.0)
    FIXED_SIZE = 328

    if len(observation) < FIXED_SIZE:
        # Дополняем нулями
        observation = np.pad(observation, (0, FIXED_SIZE - len(observation)))
    elif len(observation) > FIXED_SIZE:
        # Обрезаем до нужного размера
        observation = observation[:FIXED_SIZE]

    return observation.astype(np.float32)

if __name__ == "__main__":
    sample_data = {
        "community": {
            "pot": 150.0,
            "round_pot": 50.0,
            "small_blind": 1.0,
            "big_blind": 2.0,
            "stage_one_hot": [0, 0, 1, 0],
            "legal_moves": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        },
        "player": {
            "position": 2,
            "equity_alive": 0.90,
            "equity_2plr": 0.85,
            "equity_3plr": 0.80,
            "stack": 500.0
        },
        "stages": [
            {
                "calls": 0,  # Колла не было
                "raises": 0,  # Рейза не было
                "min_call": 1.0,  # Малый блайнд = 1
                "contribution": 1.0,  # Взнос игрока 0
                "stack_at_action": 995.0,  # Был стек 1000, после SB: 995
                "community_pot_at_action": 1.0  # Банк после SB
            },
            {
                "calls": 0,
                "raises": 0,
                "min_call": 2.0,  # BB = 2
                "contribution": 2.0,
                "stack_at_action": 998.0,  # Был стек 1000, после BB: 998
                "community_pot_at_action": 3.0  # SB + BB = 3
            },
            {
                "calls": 0,
                "raises": 0,
                "min_call": 2.0,  # BB = 2
                "contribution": 2.0,
                "stack_at_action": 998.0,  # Был стек 1000, после BB: 998
                "community_pot_at_action": 3.0  # SB + BB = 3
            }
        ]
    }
    # Инициализация агента
    observation = create_observation_space(sample_data, 2)
    print(observation,observation.shape)
    agent = DQNAgent(
        state_size=observation.shape[0],  # Размер observation вектора
        action_size=8,  # Количество возможных действий (FOLD=0 ... ALL_IN=7)
        load_model='models/dqn_fork_50stack_200epp_batch1024_weights.pth'
    )

    # Пример вызова

    action_space = [Action.FOLD, Action.CHECK, Action.RAISE_HALF_POT]  # Допустимые действия (FOLD, CHECK, CALL, RAISE_3BB)

    action = agent.action(action_space, observation, info=None)
    #print(f"Выбрано действие: {Action(action).name}")
    print(Action(action).name)
