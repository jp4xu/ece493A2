
import numpy as np
import pandas as pd
from random import randint


class rlalgorithm:
    def __init__(self, actions, learning_rate = 0.1, reward_decay = 0.9, e_greedy = 0.1, init_val = 0, name = None):
        self.initial_vals = init_val
        self.epsilon = e_greedy
        self.gamma = reward_decay
        self.alpha = learning_rate
        
        self.actions = actions
        self.num_actions = len(actions)
        self.Q = {}
        suffix = "_" + name if name else ""
        self.display_name = "Qlearning" + suffix

        # Meta
        self.episodes = 0
        self.total_r = 0
        self.max_r = float('-inf')
        self.stable_cond = 100
        self.stable_counter = 0
        self.e_decrease_factor = 0.95
        pass


    def choose_action(self, observation):
        self.check_state_exists(observation)
        """Your code goes here"""

        if np.random.uniform() >= self.epsilon:
            action = self.find_max_action(observation)
        else:
            action = np.random.choice([a for a in self.actions ])

        return action


    def learn(self, s, a, r, s_):
        self.check_state_exists(s_)
        self.total_r += r

        exp_r = self.Q[s][a]
        a_ = self.choose_action(str(s_))

        if s_ != 'terminal':
            max_a_ = self.find_max_action(s_)
            q_target = r + self.gamma * self.Q[s_][max_a_]
            self.end_episode()
        else:
            q_target = r  # next state is terminal
        self.Q[s][a] += self.alpha * (q_target - exp_r)  # update
        return s_, a_


    def end_episode(self):
        # Metadata tracking for end of episode
        self.episodes += 1
        if self.total_r > self.max_r:
            self.max_r = self.total_r
            self.stable_counter = 0
        else:
            self.stable_counter += 1
            if self.stable_counter == self.stable_cond:
                self.epsilon *= self.e_decrease_factor
                self.stable_counter = 0
            
        self.total_r = 0


    def find_max_action(self, state):
        max_reward = float('-inf')
        max_action = -1
        dic = self.Q[state]
        ind = randint(0, self.num_actions -1)
        for i in range(self.num_actions):
            ind = (ind + 1) % self.num_actions
            act = self.actions[ind]
            if dic[act] > max_reward:
                max_reward = dic[act]
                max_action = act
                
        return max_action


    def check_state_exists(self, state):
        if state not in self.Q:
            self.Q[state] = {a: self.initial_vals for a in self.actions}
