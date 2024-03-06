import numpy as np
import gymnasium as gym
import tqdm
from common.Algo_TD_SARSA import TD_SARSA


# Double Q-Learning 算法
class TD_DQLearning(TD_SARSA):
    def __init__(self, env, episodes, policy, alpha, gamma, epsilon):
        super().__init__(env, episodes, policy, alpha, gamma, epsilon)
        self.Q1 = self.Q                        # 初始化第一个Q表
        self.Q2 = np.zeros((self.nS, self.nA))  # 初始化第二个Q表

    # 两个相同价值的动作只选择第一个
    def update_policy_max(self, state): # 按e-greedy策略更新行为策略
        best_action = np.argmax(self.Q1[state] + self.Q2[state])
        self.behavior_policy[state] = self.epsilon/(self.nA-1)
        self.behavior_policy[state, best_action] = 1 - self.epsilon

    # 两个相同价值的动作具有相同的概率被选择，而不是只选择第一个
    def update_policy_average(self, state):
        Q = self.Q1[state] + self.Q2[state]
        best_actions = np.argwhere(Q == np.max(Q))
        best_actions_count = len(best_actions)
        if best_actions_count == self.nA:
            self.behavior_policy[state][:] = 1 / best_actions_count
        else:
            for action in range(self.nA):
                if action in best_actions:
                    self.behavior_policy[state][action] = (1 - self.epsilon) / best_actions_count
                else:
                    self.behavior_policy[state][action] = self.epsilon / (self.nA - best_actions_count)

    def run(self):
        id = 1  # 第一次用Q1
        for _ in tqdm.trange(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            done = False
            while not done:  # 幕内采样
                curr_action = self.choose_action(curr_state)    # 选择动作
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                # 式（10.5.1）q1(s,a) <- q1(s,a) + alpha * [r + gamma * q2(s',argmax(q1(s',:))) - q1(s,a)]
                if id == 1:
                    id = 2  # 下一次用Q2
                    best_action = np.argmax(self.Q1[next_state])
                    self.Q1[curr_state, curr_action] += self.alpha * (reward + self.gamma * self.Q2[next_state, best_action] - self.Q1[curr_state, curr_action])
                else:  # assert id == 2
                    id = 1  # 下一次用Q1
                    best_action = np.argmax(self.Q2[next_state])
                    self.Q2[curr_state, curr_action] += self.alpha * (reward + self.gamma * self.Q1[next_state, best_action] - self.Q2[curr_state, curr_action])
                self.update_policy_average(curr_state)                  # 更新行为策略
                curr_state = next_state
        return (self.Q1 + self.Q2)/2
