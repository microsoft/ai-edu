import numpy as np
import gymnasium as gym
import tqdm
from common.Algo_TD_Base import TD_Base

# 基类
class TD_SARSA(TD_Base):
    def choose_action(self, state): # 选择动作，从行为策略中按概率选择
        action = np.random.choice(self.nA, p=self.behavior_policy[state])
        return action

    def update_policy_max(self, state): # 按e-greedy策略更新行为策略
        best_action = np.argmax(self.Q[state])
        self.behavior_policy[state] = self.epsilon/(self.nA-1)
        self.behavior_policy[state, best_action] = 1 - self.epsilon
        # 以下为传统实现
        # self.behavior_policy[state] = self.epsilon/self.nA
        # self.behavior_policy[state, best_action] += 1 - self.epsilon        

    def run(self):
        for i in tqdm.trange(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            curr_action = self.choose_action(curr_state)        # 选择动作
            done = False
            while not done:  # 幕内采样
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                next_action = self.choose_action(next_state)    # 选择动作
                # 式（10.3.4）：q(s,a) <- q(s,a) + alpha * [r + gamma * q(s',a') - q(s,a)]
                self.Q[curr_state, curr_action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[curr_state, curr_action])
                self.update_policy_max(curr_state)                  # 更新行为策略
                curr_state = next_state
                curr_action = next_action
        return self.Q
