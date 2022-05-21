import numpy as np
from enum import Enum
import tqdm
import math

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

# 过程奖励值
R1 = {
    States.Start: {States.Start: 0,    States.A:0},
    States.A:     {States.Start:0,     States.B:0},
    States.B:     {States.A:0,         States.C:0},
    States.C:     {States.B:0,         States.D:0},
    States.D:     {States.C:0,         States.Home:1},
    States.Home:  {States.Home:0},
}

R2 = {
    States.Start: {States.Start:-1,    States.A:-1},
    States.A:     {States.Start:-1,    States.B:-1},
    States.B:     {States.A:-1,        States.C:-1},
    States.C:     {States.B:-1,        States.D:-1},
    States.D:     {States.C:-1,        States.Home:0},
    States.Home:  {States.Home:0},
}

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


def Matrix(dataModel, gamma):
    I = np.eye(dataModel.N) * (1+1e-7)
    #I = np.eye(dataModel.N)
    tmp1 = I - gamma * dataModel.P
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, dataModel.R)
    return vs

R3 = [
    0,  # S
    0,  # A
    0,  # B
    0,  # C
    0.5,   # D
    0,                  # Home
]
R4 = [
    (-1)*0.5+(-1)*0.5,  # S
    (-1)*0.5+(-1)*0.5,  # A
    (-1)*0.5+(-1)*0.5,  # B
    (-1)*0.5+(-1)*0.5,  # C
    (-1)*0.5+(0)*0.5,  # D
    0*1.0,                 # Home
]

if __name__=="__main__":
    dataModel = DataModel(R3)
    print("精确状态值:", Matrix(dataModel, 1))
    print("精确状态值:", Matrix(dataModel, 0.9))
