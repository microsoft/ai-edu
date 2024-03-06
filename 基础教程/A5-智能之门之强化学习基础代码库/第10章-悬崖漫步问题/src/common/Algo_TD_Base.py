import numpy as np
import gymnasium as gym
import tqdm


# 基类
class TD_Base(object):
    # 初始化，可重载
    def __init__(self, 
        env: gym.Env,           # 环境变量
        episodes: int,          # 采样幕数
        policy: np.ndarray,     # 输入策略        
        alpha: float = 0.1,     # 学习率
        gamma: float = 0.9,     # 折扣
        epsilon: float = 0.1,   # 探索率
    ):
        self.nS = env.observation_space.n       # 状态空间的大小
        self.nA = env.action_space.n            # 动作空间的大小
        self.V = np.zeros(self.nS)              # 增量计算 V
        self.Q = np.zeros((self.nS, self.nA))   # 增量计算 Q
        self.behavior_policy = policy           # 输入策略
        self.episodes = episodes                # 采样幕数
        self.env = env                          # 环境变量
        self.alpha = alpha                      # 学习率
        self.gamma = gamma                      # 折扣
        self.epsilon = epsilon                  # 探索率

    # 在子类做算法实现
    def run(self):
        raise NotImplementedError
