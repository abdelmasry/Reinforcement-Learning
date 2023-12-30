"""
######################## Reinforcement Learning Assignment#2 ########################

@abdelmasry

############################ Assignment Description #################################
In this assignment you are supposed to use another environment from Open AI gym 
classic environments, which is MountainCar-v0. 
The target of this game is to climb the hill and reach the yellow flag.
################################# Observations ######################################
Type: Box (2)
| Num | Observation |  Min  | Max  |
|:---:|:-----------:|:-----:|:----:|
|  0  |  Position   | -1.2  | 0.6  |
|  1  |  Velocity   | -0.07 | 0.07 |
###################################################################################
## Actions
### Type: Discrete (3)
| Num |   Action   |
|:---:|:----------:|
|  0  | Push Left  |
|  1  |  No Push   |
|  2  | Push Right |
###################################################################################
## Reward:
### Reward is -1 for each time step, until the goal position of 0.5 is reached. 
### There is no penalty for climbing the left hill, which upon reached acts as a wall.
###################################################################################
## StartingState:
### Random position from -0.6 to -0.4 with no velocity.
###################################################################################
## EpisodeTermination:
### The episode ends when you reach 0.5 position, or if 200 iterations are reached.
###################################################################################
## Assignment Requirements:
### 1- An implementation in python for Monte Carlo, Q_Learning and 
#### SARSA algorithms based on the MountainCar-v0 environment.
### 2- A comparison between the three algorithms in terms of accuracy and 
#### conversion time (in episodes).
###################################################################################
"""
# Importing Packages
import time
import sys
import gym
import pygame

from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Specifying the Environment
ENV = gym.make("CartPole-v1",render_mode='human')
env = gym.make("CartPole-v1")
# Explore state (observation) space
#print("State space:", env.observation_space)
#print("- low:", env.observation_space.low)
#print("- high:", env.observation_space.high)

# Explore the action space
#print("Action space:", env.action_space)


################## Discretize the State Space with a Uniform Grid #################
def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space."""
    grid = [
        np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1]
        for dim in range(len(bins))
    ]
    return grid


def discretize(sample, grid):
    """Discretize a sample as per given grid."""
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))


############################## Sampling-Based Mehods Class ##############################
class SamplingBasedMehods:
    """Sampling-Based Methods that can act on a continuous state space by discretizing it."""

    def __init__(
        self,
        env,
        state_grid,
        sampling_method,
        alpha=0.02,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay_rate=0.9995,
        min_epsilon=0.01,
        seed=505,
    ):
        # Sampling Method
        self.sampling_method = sampling_method

        # Environment Info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)
        self.action_size = self.env.action_space.n
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)

        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = (
            epsilon_decay_rate  # how quickly should we decrease epsilon
        )
        self.min_epsilon = min_epsilon

        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        self.episode = []
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode="train"):
        """Pick next action and update internal Q table"""
        if self.sampling_method == "Q_learning":
            state = self.preprocess_state(state)
            if mode == "test":
                # Test mode: Simply produce an action
                action = np.argmax(self.q_table[state])
            else:
                # Train mode (default): Update Q table, pick next action
                # Note: We update the Q table entry for the *last* (state, action) pair with the maximum Q-value of the current state
                self.q_table[self.last_state + (self.last_action,)] += self.alpha * (
                    reward
                    + self.gamma * max(self.q_table[state])
                    - self.q_table[self.last_state + (self.last_action,)]
                )
                
                # epsilon-greedy-policy
                do_exploration = np.random.uniform(0, 1) < self.epsilon
                if do_exploration:
                    # Pick a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    # Pick the best action from Q table
                    action = np.argmax(self.q_table[state])

            # Roll over current state, action for next step
            self.last_state = state
            self.last_action = action
            return action

        elif self.sampling_method == "Monte_Carlo":
            """Pick next action and store experience"""
            state = self.preprocess_state(state)
            if mode == "test":
                # Test mode: Simply produce an action
                action = np.argmax(self.q_table[state])
            else:
                # Train mode (default): Store experience, update Q table, pick next action
                self.episode.append((self.last_state, self.last_action, reward))
                # epsilon-greedy-policy
                do_exploration = np.random.uniform(0, 1) < self.epsilon
                if do_exploration:
                    # Pick a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    # Pick the best action from Q table
                    action = np.argmax(self.q_table[state])
                self.last_state = state
                self.last_action = action
                if done: 
                    # At the end of the episode, update Q values using Monte Carlo method
                    G = 0
                    for i in range(len(self.episode) - 1, -1, -1):
                        state, action, reward = self.episode[i]
                        G = self.gamma * G + reward
                        if state not in [x[0] for x in self.episode[0:i]]:
                            # First visit to this state in the episode
                            self.q_table[state][action] += self.alpha * (
                                G - self.q_table[state][action]
                            )
                    self.episode = []
            return action

        elif self.sampling_method == "SARSA":
            
            """On policy TD control method, learn action-value function"""
            state = self.preprocess_state(state)

            if mode == "test":
                # Test mode: Simply produce an action
                action = np.argmax(self.q_table[state])
            else:
                # epsilon-greedy-policy
                do_exploration = np.random.uniform(0, 1) < self.epsilon
                if do_exploration:
                    # Pick a random action
                    action = np.random.randint(0, self.action_size)
                else:
                    # Pick the best action from Q table
                    action = np.argmax(self.q_table[state])
                # Train mode (default): Update Q table, pick next action
                # chooses an action following the current policy and updates its Q-values

                if self.last_state is not None:
                    self.q_table[self.last_state][self.last_action] += self.alpha * (
                        reward
                        + self.gamma * self.q_table[state][action]
                        - self.q_table[self.last_state][self.last_action]
                    )

                # Roll over current state, action for next step
                self.last_state = state
                self.last_action = action
            return action

    def run(self, num_episodes, mode="train"):
        """Run agent in given reinforcement learning environment and return scores."""
        scores = []
        self.num_episodes = num_episodes
        max_avg_score = 0
        for i_episode in range(1, num_episodes + 1):
            # Initialize episode
            state = self.env.reset()
            action = self.reset_episode(state)
            total_reward = 0
            done = False

            while not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                action = self.act(state, reward, done, mode)

            # Save final score
            scores.append(total_reward)
            # Print episode stats
            if mode == "train":
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score
                if i_episode % 100 == 0:
                    print(
                        "\rEpisode {}/{} | Max Average Score: {}".format(
                            i_episode, num_episodes, max_avg_score
                        ),
                        end="",
                    )
                    sys.stdout.flush()
        self.accuracy = max_avg_score
        return scores

    #* DEFINE IT OUTSIDE OF THE CLASS AND ACT WITH THE AGENT, CHECK RENDER MODE IF NECESSARY *#
  


###################################################################################
# Create a grid to discretize the state space
state_grid = create_uniform_grid(
    ENV.observation_space.low, ENV.observation_space.high, bins=(6, 12, 24, 12) # GOOD CHOICE FOR REWARD
)

def test(env,agent):
    """Testing the sampling method by rendering the enviroment"""

    state = env.reset()
    score = 0
    
    for t in range(1000):
        action = agent.act(state, mode="test")
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
    print("Final score:", score)
    env.close()
################################### Monte Carlo ###################################
#MONTE_CARLO_AGENT = SamplingBasedMehods(env, state_grid, sampling_method="Monte_Carlo")
#MONTE_CARLO_AGENT.run(num_episodes=60000)
# MONTE_CARLO_AGENT.test()
################################### Q-Learning ####################################
#Q_AGENT = SamplingBasedMehods(env, state_grid, sampling_method="Q_learning")
#Q_AGENT.run(num_episodes=40000)
#test(env=ENV,agent=Q_AGENT)

#################################### SARSA ########################################
#SARSA_AGENT = SamplingBasedMehods(env, state_grid, sampling_method="SARSA")
#SARSA_AGENT.run(num_episodes=20000)
# SARSA_AGENT.test()
###################################################################################
# abdelmasry #
