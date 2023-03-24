import numpy as np
from enum import Enum
import Algo_Sampling as algo

# 状态定义
class States(Enum):
    Start = 0
    A = 1
    B = 2
    C = 3
    D = 4
    Home = 5
    End = 6    # 终止状态

# 状态转移矩阵
P = np.array(
    [  # S    A    B    C    D    Home End    
        [0.5, 0.5, 0,   0,   0,   0,    0], # S
        [0.5, 0,   0.5, 0,   0,   0,    0], # A
        [0,   0.5, 0,   0.5, 0,   0,    0], # B
        [0,   0,   0.5, 0,   0.5, 0,    0], # C
        [0,   0,   0,   0.5, 0,   0.5,  0], # D
        [0,   0,   0,   0,   0,   0,    1],  # Home
        [0,   0,   0,   0,   0,   0,    1]  # End
])

#   Start, A,   B,  C,  D, H, End
R1 = [0,   0,  0,  0,  0, 1, 0]
#R2 = [-1, -1, -1, -1, -1, 0, 0]

class DataModel(object):
    def __init__(self, R):
        self.P = P                          # 状态转移矩阵
        self.R = R                          # 奖励
        self.S = States                     # 状态集
        self.nS = len(self.S)               # 状态数量
        self.end_states = [self.S.End]      # 终止状态集

    # 判断给定状态是否为终止状态
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    def get_reward(self, s):
        return self.R[s.value]

    # 根据转移概率前进一步，返回（下一个状态、即时奖励、是否为终止）
    def step(self, s):
        next_s = np.random.choice(self.S, p=self.P[s.value])
        return next_s, self.get_reward(s), self.is_end(next_s)


if __name__=="__main__":
    episodes = 1000
    print("奖励值:", R1)
    model = DataModel(R1)
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))
        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            V[s.value] = algo.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        algo.print_V(model, V)
