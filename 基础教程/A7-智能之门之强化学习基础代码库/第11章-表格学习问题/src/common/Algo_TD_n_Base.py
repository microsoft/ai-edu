import numpy as np
import gymnasium as gym
import tqdm
from common.Algo_TD_Base import TD_Base

# 基类
class TD_n_Base(TD_Base):
    def __init__(self, 
        env: gym.Env,           # 环境变量
        episodes: int,          # 采样幕数
        policy: np.ndarray,     # 输入策略        
        alpha: float = 0.1,     # 学习率
        gamma: float = 0.9,     # 折扣
        epsilon: float = 0.1,    # epsilon-greedy
        n: int = 1,             # n-step TD
    ):
        super().__init__(env, episodes, policy, alpha=alpha, gamma=gamma, epsilon=epsilon)
        self.n = n                              # n-step TD

    # 在子类做算法实现
    def run(self):
        raise NotImplementedError
