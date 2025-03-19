import torch
import numpy as np
import json
from gym_env.enums import Action, Stage


class DQNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PokerAgent:
    def __init__(self, model_weights_path, model_json_path):
        with open(model_json_path) as f:
            model_config = json.load(f)

        self.state_size = model_config["state_size"]  # 328
        self.action_size = model_config["action_size"]  # 8

        self.model = DQNetwork(self.state_size, self.action_size)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

        self.action_mapping = {
            0: Action.FOLD,
            1: Action.CHECK,
            2: Action.CALL,
            3: Action.RAISE_3BB,
            4: Action.RAISE_HALF_POT,
            5: Action.RAISE_POT,
            6: Action.RAISE_2POT,
            7: Action.ALL_IN
        }

    def _vectorize_data(self, opencv_data):
        # Нормализация
        bb_norm = 2 * 100  # big_blind=2 из JSON

        # Player Data
        player_data = [
            opencv_data['player_position'],
            opencv_data['player_stack'] / bb_norm,
            0.0  # equity (если не используется)
        ]

        # Community Data
        stage_encoding = np.zeros(4)
        stage = opencv_data['stage']
        if stage == Stage.PREFLOP:
            stage_encoding[0] = 1
        elif stage == Stage.FLOP:
            stage_encoding[1] = 1
        elif stage == Stage.TURN:
            stage_encoding[2] = 1
        elif stage == Stage.RIVER:
            stage_encoding[3] = 1

        community_data = [
            *stage_encoding.tolist(),
            opencv_data['community_pot'] / bb_norm,
            opencv_data['current_round_pot'] / bb_norm,
            1 / bb_norm,  # small_blind=1
            2 / bb_norm,  # big_blind=2
            *[1 if action in opencv_data['legal_moves'] else 0 for action in self.action_mapping.values()]
        ]

        # Stage Data (заглушка для 328 элементов)
        stage_data = [0.0] * (328 - len(player_data) - len(community_data))

        return np.concatenate([player_data, community_data, stage_data])

    def predict_action(self, opencv_data):
        observation = self._vectorize_data(opencv_data)
        print(observation)
        state = torch.FloatTensor(observation).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state).squeeze().numpy()

        legal_actions = [a.value for a in opencv_data['legal_moves']]
        masked_q = np.full(self.action_size, -np.inf)
        for action in legal_actions:
            if action in self.action_mapping:
                masked_q[action] = q_values[action]

        return self.action_mapping[np.argmax(masked_q)]


# Пример использования
if __name__ == "__main__":
    agent = PokerAgent(
        model_weights_path="models/dqn_fork_50stack_200ep_weights.pth",
        model_json_path="models/dqn_fork_50stack_200ep_json.json"
    )

    opencv_data = {
        'player_position': 0,
        'player_stack': 1000,
        'community_pot': 150,
        'current_round_pot': 50,
        'stage': Stage.FLOP,
        'legal_moves': [Action.FOLD, Action.CALL, Action.RAISE_3BB]
    }

    action = agent.predict_action(opencv_data)
    print(f"Action: {action.name}")