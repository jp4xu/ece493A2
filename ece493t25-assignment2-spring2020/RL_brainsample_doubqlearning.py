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
        self.Q1 = {}
        self.Q2 = {}
        suffix = "_" + name if name else ""
        self.display_name = "Double_Qlearning" + suffix

        # Meta
        self.episodes = 0
        self.total_r = 0
        self.max_r = float('-inf')
        self.stable_cond = 100
        self.stable_counter = 0
        self.e_decrease_factor = 1
        pass

    def choose_action(self, observation):
        self.check_state_exists(observation)

        if np.random.uniform() >= self.epsilon:
            action = self.find_max_action(observation, 2)
        else:
            action = np.random.choice([a for a in self.actions ])
        
        return action


    def learn(self, s, a, r, s_):
        self.check_state_exists(s_)
        self.total_r += r
        
        a_ = self.choose_action(str(s_))

        update_mode = randint(0,1)
        # Set table as opposite of mode
        table = self.Q1 if update_mode else self.Q2
        exp_r = table[s][a]
        if s_ != 'terminal':
            max_a_ = self.find_max_action(s_, update_mode)
            q_target = r + self.gamma * table[s_][max_a_]
            
            self.end_episode
        else:
            q_target = r  # next state is terminal

        table[s][a] += self.alpha * (q_target - exp_r)  # update
        
        return s_, a_


    def end_episode(self):
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
        

    def find_max_action(self, state, m):
        max_reward = float('-inf')
        max_action = -1
        ind = randint(0, self.num_actions -1)
        for i in range(self.num_actions):
            ind = (ind + 1) % self.num_actions
            act = self.actions[ind]
            val = self.max_eval(state,act, m)
            if val > max_reward:
                max_reward = val
                max_action = act
                
        return max_action

    def max_eval(self, state, action, m):
        if m == 0:
            return self.Q1[state][action]
        elif m == 1:
            return self.Q2[state][action]
        elif m == 2:
            return self.Q2[state][action] + self.Q1[state][action]

    def check_state_exists(self, state):
        if state not in self.Q1:
            self.Q1[state] = {a: self.initial_vals for a in self.actions}

        if state not in self.Q2:
            self.Q2[state] = {a: self.initial_vals for a in self.actions}
