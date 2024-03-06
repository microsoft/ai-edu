import numpy as np
import gymnasium as gym
import tqdm
from common.Algo_TD_SARSA import TD_SARSA

# Q-Learning 算法
class TD_QLearning(TD_SARSA):
    def run(self):
        for _ in tqdm.trange(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            done = False
            while not done:  # 幕内采样
                curr_action = self.choose_action(curr_state)    # 选择动作
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                # 式（10.4.1）q(s,a) <- q(s,a) + alpha * [r + gamma * Max[q(s')] - q(s,a)]
                self.Q[curr_state, curr_action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[curr_state, curr_action])
                self.update_policy_max(curr_state)                  # 更新行为策略
                curr_state = next_state
        return self.Q

