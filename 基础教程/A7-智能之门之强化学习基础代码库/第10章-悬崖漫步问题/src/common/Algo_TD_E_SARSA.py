import numpy as np
import gymnasium as gym
import tqdm
from common.Algo_TD_SARSA import TD_SARSA

# 期望SARSA
class TD_E_SARSA(TD_SARSA):
    def expected_Q(self, state): # 计算期望 Q
        v = np.dot(self.Q[state], self.behavior_policy[state])
        return v

    # E_SARSA 算法
    def run(self):
        for i in tqdm.trange(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            curr_action = self.choose_action(curr_state)        # 选择动作
            done = False
            step = 0
            while not done:  # 幕内采样
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                next_action = self.choose_action(next_state)    # 选择动作
                # 式（10.4.1）q(s,a) <- q(s,a) + alpha * [r + gamma * q(s',a') - q(s,a)]
                expected = self.expected_Q(next_state)
                self.Q[curr_state, curr_action] += self.alpha * (reward + self.gamma * expected - self.Q[curr_state, curr_action])
                self.update_policy_max(curr_state)                  # 更新行为策略
                curr_state = next_state
                curr_action = next_action
        return self.Q
