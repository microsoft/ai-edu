import numpy as np
from enum import Enum

class States(Enum):
    Start = 0
    A = 1
    B = 2
    C = 3
    D = 4
    Home = 5

P = np.array(
    [  # S    A    B    C    D    H       
        [0.5, 0.5, 0,   0,   0  , 0  ], # S
        [0.5, 0,   0.5, 0,   0,   0, ], # A
        [0,   0.5, 0,   0.5, 0,   0, ], # B
        [0,   0,   0.5, 0,   0.5, 0, ], # C
        [0,   0,   0,   0.5, 0,   0.5,], # D
        [0,   0,   0,   0,   0,   1], # Home
])

class DataModel(object):
    def __init__(self, R):
        self.P = P                          # 状态转移矩阵
        self.R = R                          # 奖励
        self.S = States                     # 状态集
        self.nS = len(self.S)                # 状态数量
        self.end_states = [self.S.Home]      # 终止状态集

    # 判断给定状态是否为终止状态
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    def get_reward(self, s, s_next):
        return self.R[s][s_next]

    # 根据转移概率前进一步，返回（下一个状态、即时奖励、是否为终止）
    def step(self, s):
        s_next = np.random.choice(self.S, p=self.P[s.value])
        return s_next, self.get_reward(s, s_next), self.is_end(s_next)
