import numpy as np
from gym_env.env import HoldemTable, PlayerShell
from gym_env.enums import Action, Stage
from Testplay import DQNAgent, create_observation_space, Action


class HumanAgent:
    """Класс-заглушка для человеческого игрока"""

    def __init__(self, name="Human"):
        self.name = name
        self.autoplay = False  # важно для работы цикла

    def action(self, legal_moves, observation, info):
        return self.choose_action(legal_moves)

    def choose_action(self, legal_moves):
        """Консольный интерфейс для выбора действия"""
        print("\nДоступные действия:")
        for i, action in enumerate(legal_moves):
            print(f"{i}: {Action(action).name}")

        while True:
            try:
                choice = int(input("Ваш выбор (введите номер): "))
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice].value
                print("Некорректный номер. Попробуйте снова.")
            except ValueError:
                print("Введите число!")


def print_game_state(env, human_player_id):
    """Вывод текущего состояния игры"""
    print("\n" + "=" * 40)
    print(f"Этап: {env.stage.name}")
    print(f"Карты на столе: {env.table_cards}")

    for i, player in enumerate(env.players):
        if i == human_player_id:
            print(f"\nВаши карты: {player.cards}")
            print(f"Ваш стек: {player.stack}")
        else:
            print(f"\nИгрок {i} ({player.name}):")
            print(f"Стек: {player.stack}")
            print(f"Последнее действие: {player.last_action_in_stage}")

    print(f"\nОбщий банк: {env.community_pot + env.current_round_pot}")
    print(f"Текущий раунд: {env.current_round_pot}")


def main():
    # Инициализация окружения
    env = HoldemTable(
        initial_stacks=1000,
        small_blind=10,
        big_blind=20,
        render=False,
        calculate_equity=False
    )

    # Инициализация агентов
    human_agent = HumanAgent()
    dqn_agent = DQNAgent(
        state_size=328,
        action_size=8,
        load_model='models/dqn_fork_50stack_200epp_batch1024_weights.pth'
    )

    # Добавление игроков
    env.add_player(dqn_agent)  # AI игрок
    env.add_player(human_agent)  # Человек

    human_player_id = 1  # ID человеческого игрока
    env.reset()

    while True:
        current_player_id = env.player_cycle.idx
        current_player = env.players[current_player_id]

        if env.done:
            print(f"\nИгра окончена! Победитель: {env.players[env.winner_ix].name}")
            break

        # Вывод состояния для человека
        if current_player_id == human_player_id:
            print_game_state(env, human_player_id)

        # Получение действия
        if current_player.name == "Human":
            action = human_agent.choose_action(env.legal_moves)
        else:
            # Подготовка данных для модели
            observation = create_observation_space({
                "community": {
                    "pot": env.community_pot,
                    "round_pot": env.current_round_pot,
                    "small_blind": env.small_blind,
                    "big_blind": env.big_blind,
                    "stage_one_hot": env.community_data.stage,
                    "legal_moves": [int(m in env.legal_moves) for m in Action]
                },
                "player": {
                    "position": current_player_id,
                    "equity_alive": env.player_data.equity_to_river_alive,
                    "equity_2plr": env.player_data.equity_to_river_2plr,
                    "equity_3plr": env.player_data.equity_to_river_3plr,
                    "stack": current_player.stack
                },
                "stages": [
                    {
                        "calls": stage.calls[current_player_id],
                        "raises": stage.raises[current_player_id],
                        "min_call": stage.min_call_at_action[current_player_id],
                        "contribution": stage.contribution[current_player_id],
                        "stack_at_action": stage.stack_at_action[current_player_id],
                        "community_pot_at_action": stage.community_pot_at_action[current_player_id]
                    } for stage in env.stage_data
                ]
            }, env.big_blind)

            action = dqn_agent.action(env.legal_moves, observation, None)

        # Выполнение действия
        env.step(action)
        print(f"\n{current_player.name} выполнил: {Action(action).name}")


if __name__ == "__main__":
    main()