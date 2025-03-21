"""
neuron poker

Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
  main.py learn_table_scraping [options]
  main.py selfplay ppo_train [options]
  main.py selfplay mcts_agent [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500].

"""

import logging

import gym
import numpy as np
import pandas as pd
from docopt import docopt

from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


# pylint: disable=import-outside-toplevel

def command_line_parser():
    """Entry function"""
    args = docopt(__doc__)
    if args['--log']:
        logfile = args['--log']
    else:
        print("Using default log file")
        logfile = 'default'
    model_name = args['--name'] if args['--name'] else 'dqn1'
    screenloglevel = logging.INFO if not args['--screenloglevel'] else \
        getattr(logging, args['--screenloglevel'].upper())
    _ = get_config()
    init_logger(screenlevel=screenloglevel, filename=logfile)
    print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

    if args['selfplay']:
        num_episodes = 1 if not args['--episodes'] else int(args['--episodes'])
        runner = SelfPlay(render=args['--render'], num_episodes=num_episodes,
                          use_cpp_montecarlo=args['--use_cpp_montecarlo'],
                          funds_plot=args['--funds_plot'],
                          stack=int(args['--stack']))

        if args['random']:
            runner.random_agents()

        elif args['keypress']:
            runner.key_press_agents()

        elif args['consider_equity']:
            runner.equity_vs_random()

        elif args['equity_improvement']:
            improvement_rounds = int(args['--improvement_rounds'])
            runner.equity_self_improvement(improvement_rounds)

        elif args['dqn_train']:
            runner.dqn_train_keras_rl(model_name)

        elif args['dqn_play']:
            runner.dqn_play_keras_rl(model_name)
        
        elif args['ppo_train']:
            runner.ppo_train_torch_rl()
        
        elif args['mcts_agent']:
            runner.mcts_agent()

    else:
        raise RuntimeError("Argument not yet implemented")


class SelfPlay:
    """Orchestration of playing against itself"""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=5):
        """Initialize"""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack
        self.log = logging.getLogger(__name__)

    def random_agents(self):
        """Create an environment with 6 random players"""
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2
        # self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env = gym.make(env_name, initial_stacks=5, render=self.render)
        for _ in range(num_of_plrs):
            player = RandomPlayer()
            self.env.add_player(player)
        # print(self.env.observation_space.shape[0])
        self.env.reset()

    def key_press_agents(self):
        """Create an environment with 6 key press agents"""
        from agents.agent_keypress import Player as KeyPressAgent
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        for _ in range(num_of_plrs):
            player = KeyPressAgent()
            self.env.add_player(player)

        self.env.reset()

    def equity_vs_random(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        self.env.add_player(EquityPlayer(name='equity/80/80', min_call_equity=.8, min_bet_equity=-.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def equity_self_improvement(self, improvement_rounds):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        calling = [.1, .2, .3, .4, .5, .6]
        betting = [.2, .3, .4, .5, .6, .7]

        for improvement_round in range(improvement_rounds):
            env_name = 'neuron_poker-v0'
            self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
            for i in range(6):
                self.env.add_player(EquityPlayer(name=f'Equity/{calling[i]}/{betting[i]}',
                                                 min_call_equity=calling[i],
                                                 min_bet_equity=betting[i]))

            for _ in range(self.num_episodes):
                self.env.reset()
                self.winner_in_episodes.append(self.env.winner_ix)

            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = int(league_table.index[0])
            print(league_table)
            print(f"Best Player: {best_player}")

            # self improve:
            self.log.info(f"Self improvment round {improvement_round}")
            for i in range(6):
                calling[i] = np.mean([calling[i], calling[best_player]])
                self.log.info(f"New calling for player {i} is {calling[i]}")
                betting[i] = np.mean([betting[i], betting[best_player]])
                self.log.info(f"New betting for player {i} is {betting[i]}")

    def dqn_train_keras_rl(self, model_name):
        """Implementation of kreras-rl deep q learing."""
        from agents.agent_consider_equity import Player as EquityPlayer
        # from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_torch_dqn import Player as DQNPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=20, funds_plot=self.funds_plot, render=self.render,
                       use_cpp_montecarlo=self.use_cpp_montecarlo)

        np.random.seed(123)
        self.env.seed(123)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
        self.env.add_player(RandomPlayer())
        #self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
        self.env.add_player(PlayerShell(name=model_name, stack_size=10))
        #self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        self.env.add_player(EquityPlayer(name='equity/60/40', min_call_equity=.6, min_bet_equity=.4))
        self.env.add_player(RandomPlayer())
        # self.env.add_player(RandomPlayer())
        # self.env.add_player(RandomPlayer())
        # self.env.add_player(RandomPlayer())
        # env.add_player(PlayerShell(name='keras-rl', stack_size=self.stack))  # shell is used for callback to keras rl
        self.env.reset()

        dqn = DQNPlayer(env=self.env)
        # dqn.initiate_agent(env)
        dqn.train(env_name=model_name)
    
    def ppo_train_torch_rl(self):
        '''Implementation of torch-rl proximal policy optimization'''
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_ppo import Player as PPOPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=10, funds_plot=self.funds_plot, render=self.render,
                          use_cpp_montecarlo=self.use_cpp_montecarlo)
        np.random.seed(123)
        self.env.seed(123)
        # self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
        # self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=.8))
        # self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
        # self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(PlayerShell(name='torch-ppo', stack_size=10))
        self.env.reset()
        ppo = PPOPlayer(env=self.env, action_size=self.env.action_space.n, state_size=self.env.observation_space[0])
        ppo.train()

    def dqn_play_keras_rl(self, model_name):
        """Create 6 players, one of them a trained DQN"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_torch_dqn import Player as DQNPlayer
        from agents.Human_agent import Player as Human
        from agents.AI_play import Player as AI
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=300, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
        #self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=.8))
        self.env.add_player(AI(name='AI_2', load_model='fork_50stack_200epp_batch1024_0001sp'))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
        self.env.add_player(AI(name='AI_3', load_model='fork_50stack_200epp_new_reward_calc'))
        #self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        #self.env.add_player(RandomPlayer())
        self.env.add_player(Human())
        self.env.add_player(PlayerShell(name='AI_1', stack_size=self.stack))

        self.env.reset()

        dqn = DQNPlayer(load_model=model_name, env=self.env)
        dqn.play(nb_episodes=self.num_episodes, render=self.render)

    def dqn_train_custom_q1(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_custom_q1 import Player as Custom_Q1
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        # self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        # self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        # self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(Custom_Q1(name='Deep_Q1'))

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")
    
    def mcts_agent(self, mode='train', num_episodes=20, num_runs=5):
        """Run MCTS agent for multiple runs and plot rewards for each run."""
        import os
        import time
        import torch
        import gym
        import matplotlib.pyplot as plt
        import pandas as pd
        from agents.agent_random import Player as RandomPlayer
        from agents.agent_MCTS import Player as MCTSPlayer
        from agents.agent_MCTS import MCTSNet, optim, train_neural_network, deque
        from agents.agent_consider_equity import Player as EquityPlayer
        env_name = 'neuron_poker-v0'
        rewards_per_run = []  # Store rewards for each run
    
        for run in range(num_runs):
            print(f"Starting Run {run + 1}/{num_runs}")
            self.env = gym.make(env_name, initial_stacks=10, render=self.render)
    
            # Add players
            self.env.add_player(PlayerShell(name='mcts-player', stack_size=10))
            self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
            self.env.add_player(EquityPlayer(name='equity/80/80', min_call_equity=.8, min_bet_equity=.8))
            self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
            self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
            self.env.add_player(RandomPlayer())
            self.env.reset()
    
            try:
                input_size = self.env.observation_space[0]
                num_actions = self.env.action_space.n
            except AttributeError as e:
                print(f"Error accessing observation or action space: {e}")
                return
    
            # Initialize the neural network
            neural_net = MCTSNet(input_size, num_actions)
            model_path = os.path.join('agents', 'mcts_net.pth')
    
            optimizer = optim.Adam(neural_net.parameters(), lr=0.001)
            replay_buffer = deque(maxlen=1000000)  # Store all episodes, size can be adjusted
    
            mcts_player = MCTSPlayer(self.env, neural_net, replay_buffer)
    
            # For tracking metrics (Reward, Steps/Episode)
            rewards = []
            steps = []
    
            episode = 0
            while episode < num_episodes:
                print(f"Run {run + 1}, Episode {episode + 1}/{num_episodes} - Starting...")
                state = self.env.reset()
                done = False
                steps_episode = 0
                total_reward = 0
    
                while not done:
                    action_space = self.env.legal_moves
                    action = mcts_player.action(action_space, state, None)
                    buffer, next_state, reward, done, _ = self.env.mcts_step(action)
                    for i in range(len(buffer)):
                        replay_buffer.append(buffer[i])
                    state = next_state
                    total_reward = reward
                    steps_episode += 1
    
                # Train the neural network after each episode
                print("REPLAY BUFFER IS SIZE: ", len(replay_buffer))
                train_neural_network(neural_net, replay_buffer, optimizer, batch_size=32, gamma=0.995)
    
                # Save the model after every episode
                torch.save(neural_net.state_dict(), model_path)
    
                # Append the results for plotting later
                rewards.append(total_reward)
                steps.append(steps_episode)
    
                # Log the winner of the episode
                if self.env.winner_ix is not None:
                    self.winner_in_episodes.append(self.env.winner_ix)
                else:
                    print(f"Hand did not have a winner for episode {episode}.")
                
                episode += 1
    
            # After each run, store rewards for the run
            rewards_per_run.append(rewards)
    
        # After all runs, plot the results
        self.plot_training_metrics(rewards_per_run)
    
    def plot_training_metrics(self, rewards_per_run):
        """Plot training metrics such as reward and steps for each run."""
        
        plt.figure(figsize=(10, 5))
    
        # Plot rewards per episode for each run
        for i, rewards in enumerate(rewards_per_run):
            plt.plot(rewards, label=f"Run {i + 1}")
    
        plt.title("Total Reward per Episode (Multiple Runs)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        
        # Display the plot
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    command_line_parser()