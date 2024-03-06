import numpy as np
import gymnasium as gym
import tqdm
from common.Algo_TD_Base import TD_Base

# 基类
class TD_TD0(TD_Base):
    # TD(0) 算法
    def run(self):
        for _ in tqdm.trange(self.episodes):
            curr_state, _ = self.env.reset()
            done = False
            while not done:
                action = np.random.choice(self.env.action_space.n, p=self.behavior_policy[curr_state])
                next_state, reward, done, truncated, info = self.env.step(action)
                # 式（10.2.5）：v(s) <- v(s) + alpha * [r + gamma * v(s') - v(s)]
                self.V[curr_state] += self.alpha * (reward + self.gamma * self.V[next_state] - self.V[curr_state])
                curr_state = next_state
        return self.V
