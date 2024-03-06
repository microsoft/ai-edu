import numpy as np
import gymnasium as gym


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

    # 两个相同价值的动作具有相同的概率被选择，而不是只选择第一个
    def update_policy_average(self, state):
        best_actions = np.argwhere(self.Q[state] == np.max(self.Q[state]))
        best_actions_count = len(best_actions)
        if best_actions_count == self.nA:
            self.behavior_policy[state][:] = 1 / best_actions_count
        else:
            for action in range(self.nA):
                if action in best_actions:
                    self.behavior_policy[state][action] = (1 - self.epsilon) / best_actions_count
                else:
                    self.behavior_policy[state][action] = self.epsilon / (self.nA - best_actions_count)

    # 在子类做算法实现
    def run(self):
        raise NotImplementedError
