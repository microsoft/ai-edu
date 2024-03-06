
import numpy as np
import gymnasium as gym
from common.Algo_MC_OnPolicy_Base import MC_On_Policy_Base


class MC_FirstVisit_Control_Greedy(MC_On_Policy_Base):
    def __init__(self, env: gym.Env, episodes: int, gamma: float, policy: np.ndarray):
        super().__init__(env, episodes, gamma, policy)
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float64)       # Q 值 

    def set_greedy_fun(self, greedy_fun, epsilon: float=0.1):
        self.greedy_fun = greedy_fun
        self.epsilon = epsilon

    def calculate(self):
        num_step = super().calculate()
        G = 0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)
            if not ((s,a) in self.Trajectory_sa[0:t]):  # 首次访问型
                super().increase_Cum_Count(s, a, G)     # 值累加，数量加 1
                if self.CumQ[s,a] == 0: # 是 0 时，可以避免被迫做策略更新，前提是动作价值函数大于 0
                    continue            # 当动作价值函数有正有负时，不能用这个方法
                self.Q[s,a] = self.CumQ[s,a] / self.CntQ[s,a]
                if self.greedy_fun == MC_Greedy:
                    self.policy = self.greedy_fun(self.policy, self.Q, s, self.nA)
                elif self.greedy_fun == MC_Soft_Greedy:
                    self.policy = self.greedy_fun(self.policy, self.Q, s, self.nA, self.epsilon)
                else:
                    raise Exception("未知的贪心方法")        

    def calculate_V(self):
        V, Q = super().calculate_V()
        return V, Q, self.policy
# end class


# 贪心策略
def MC_Greedy(
    policy: np.ndarray,
    Q: np.ndarray,
    s: int,
    nA: int
) -> np.ndarray:
    best_actions = np.argwhere(Q[s] == np.max(Q[s]))
    best_actions_count = len(best_actions)
    policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(nA)]
    return policy    


# 软性策略 e-soft
def MC_Soft_Greedy(
    policy: np.ndarray,
    Q: np.ndarray, 
    s: int, 
    nA: int, 
    epsilon: float
) -> np.ndarray:
    best_action = np.argmax(Q[s])
    policy[s] = epsilon/nA
    policy[s][best_action] += 1 - epsilon
    # policy[s] = [1-epsilon+epsilon/nA if a == best_action else epsilon/nA for a in range(nA)]
    return policy
