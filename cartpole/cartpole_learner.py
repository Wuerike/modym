import json
import math
import random

import numpy as np
import gymnasium as gym
import pandas as pd

from time import sleep, time
from datetime import datetime
from gymnasium.envs.registration import register

class CartPoleLearner():
    def __init__(self):
        # Define the environment
        f = open ('config.json', "r")
        config = json.loads(f.read())
        env_name = "CartPoleEnv-v0"
        register(
            id=env_name,
            entry_point='cartpole:CartPoleEnv',
            kwargs={'config':config, 'log_level': 'INFO'}
        )
        self.env = gym.make(env_name)

        # Reset environment to a random state
        self.env.reset() 

        # Discretize the environment
        self.env_bins = {
            "x": self._get_bins(-2.4, 2.4, 10),
            "x_dot": self._get_bins(-1, 1, 10),
            "theta": self._get_bins(78/180*math.pi, 102/180*math.pi, 10),
            "theta_dot": self._get_bins(-2, 2, 10),
        }

        # Qtable initialization
        self.q_table = np.zeros([10 ** self.env.observation_space.shape[0], self.env.action_space.n])

    def train(self):
        learning_rate = 0.4
        discount_factor = 1
        exploration_rate = 0.2
        exploration_decay_rate=0.99995

        episodes = 1000
        max_steps = 300
        episodes_steps = []

        print("\nTraining...")
        start = time()
        for i in range(0, episodes):
            state = self.env.reset()[0]
            steps, done = 0, False
            
            while (not done) and (steps < max_steps):
                self.env.render()

                if random.uniform(0, 1) < exploration_rate:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q_table[self._get_state_index(state)]) # Exploit learned values

                next_state, reward, done, _, _ = self.env.step(action)

                if (not done) and (steps == max_steps-1):
                    reward = 100

                old_value = self.q_table[self._get_state_index(state), action]
                next_max = np.max(self.q_table[self._get_state_index(next_state)])
                
                new_value = (1 - learning_rate) * old_value + learning_rate  * (reward + discount_factor * next_max)
                self.q_table[self._get_state_index(state), action] = new_value

                exploration_rate *= exploration_decay_rate
                state = next_state
                steps += 1
            
            print(f"Episode {i} - Steps {steps}")
            episodes_steps = np.append(episodes_steps, steps)

        execution_time = time() - start
        print(f"\nTraining finished after {execution_time} seconds")
        print(f"Avg episode performance {episodes_steps.mean()} {chr(177)} {episodes_steps.std()}")
        print(f"Max episode performance {episodes_steps.max()}")

    def test(self):

        episodes = 100
        max_steps = 600
        episodes_steps = []

        print("\nTesting...")
        start = time()
        for i in range(0, episodes):
            state = self.env.reset()[0]
            steps, done = 0, False
            
            while (not done) and (steps < max_steps):
                self.env.render()

                action = np.argmax(self.q_table[self._get_state_index(state)])
                next_state, _, done, _, _ = self.env.step(action)
               
                state = next_state
                steps += 1
            
            print(f"Episode {i} - Steps {steps}")
            episodes_steps = np.append(episodes_steps, steps)

        execution_time = time() - start
        print(f"\nTest finished after {execution_time} seconds")
        print(f"Avg episode performance {episodes_steps.mean()} {chr(177)} {episodes_steps.std()}")
        print(f"Max episode performance {episodes_steps.max()}")
        self._save_policy()

    def from_policy_test(self, path):
        try:
            df = pd.read_csv(path)
        except:
            print("Not a valid csv file")
            return
        
        self.q_table = df.to_numpy()
        self.test()

    def random_test(self):
        episodes = 10
        max_steps = 100

        for ep in range(episodes):
            initial_state = self.env.reset()
            for step in range(max_steps):
                self.env.render()
                random_action = self.env.action_space.sample()
                state, reward, done, _, _ = self.env.step(random_action)
                sleep(0.01)
                if done:
                    break

    def _get_bins(self, lower_bound, upper_bound, n_bins):
        return np.linspace(lower_bound, upper_bound, n_bins + 1)[1:-1]

    def _get_state_index(self, state):
        bins = [
            np.digitize([state[0]], self.env_bins['x'])[0],
            np.digitize([state[1]], self.env_bins['x_dot'])[0],
            np.digitize([state[2]], self.env_bins['theta'])[0],
            np.digitize([state[3]], self.env_bins['theta_dot'])[0],
        ]

        state = int("".join(map(lambda state_bin: str(state_bin), bins)))
        return state

    def _save_policy(self):
        df = pd.DataFrame(self.q_table)
        df.to_csv(f'./policies/policy_{datetime.now().strftime("%y%m%d_%H%M%S")}.csv', index=False)
