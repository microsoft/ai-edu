import numpy as np
from enum import Enum

class States(Enum):
    Start = 0
    A = 1
    B = 2
    C = 3
    D = 4
    Home = 5
    End = 6

P = np.array(
    [  # S    A    B    C    D    H    end   
        [0.5, 0.5, 0,   0,   0  , 0  , 0  ], # S
        [0.5, 0,   0.5, 0,   0,   0,   0  ], # A
        [0,   0.5, 0,   0.5, 0,   0,   0  ], # B
        [0,   0,   0.5, 0,   0.5, 0,   0  ], # C
        [0,   0,   0,   0.5, 0,   0.5, 0  ], # D
        [0,   0,   0,   0,   0,   0,   1.0], # Home
        [0,   0,   0,   0,   0,   0,   1  ], # end
])

# 状态奖励值
#     S, A, B, C, D, H, e
R1 = [0, 0, 0, 0, 0, 1, 0]

# 过程奖励值
R2 = [-1, -1, -1, -1, -1, 0, 0]

class DataModel(object):
    def __init__(self):
        self.P = P                          # 状态转移矩阵
        self.R = R1                         # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)                # 状态数量
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
        return next_s, self.get_reward(next_s), self.is_end(next_s)


def Matrix(dataModel, gamma):
    num_state = dataModel.P.shape[0]
    I = np.eye(dataModel.N) * (1+1e-7)
    #I = np.eye(dataModel.num_states)
    tmp1 = I - gamma * dataModel.P
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, dataModel.R)
    return vs

if __name__=="__main__":
    dataModel = DataModel()
    v = Matrix(dataModel, 1)
    print(np.around(v,2))
