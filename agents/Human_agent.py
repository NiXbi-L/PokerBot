"""Random player"""
import random

from gym_env.enums import Action


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='Human'):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = False

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT, Action.ALL_IN, Action.BIG_BLIND, Action.SMALL_BLIND}
        possible_moves = this_player_action_space.intersection(set(action_space))
        possible_moves = list(possible_moves)
        #action = random.choice(list(possible_moves))
        for i, actions in enumerate(possible_moves):
            print(i, Action(actions).name)
        action = possible_moves[0]
        while True:
            try:
                action = possible_moves[int(input('Выберите действие: '))]
                break
            except:
                continue
        return action
