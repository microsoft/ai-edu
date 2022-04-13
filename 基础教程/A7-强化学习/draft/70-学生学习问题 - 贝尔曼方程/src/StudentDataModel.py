
import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Game = 0
    Class1 = 1
    Class2 = 2
    Class3 = 3
    Pass = 4
    Rest = 5
    End = 6

# 奖励向量
# [Game, Class1, Class2, Class3, Pass, Rest, End]
Rewards = [-1, -2, -2, -2, 10, 1, 0]
#Rewards = [-2, 0.5, 1, 1.5, 10, -2, 0]

# 状态转移概率
P = np.array(
    [   #Game Cl1  Cl2  Cl3  Pass Rest End
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    ]
)

ground_truth = [-22.543, -12.543, 1.457, 4.321, 10.00, 0.803, 0.00]
a = [-22.5, -12.5, 1.5, 4.3, 10, 0.8, 0.00]

def RMSE(a,b):
    err = np.sqrt(np.sum(np.square(a - b))/a.shape[0])
    return err

class DataModel(object):
    def __init__(self):
        self.P = P                          # 状态转移矩阵
        self.R = Rewards                    # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)       # 状态数量
        self.end_states = [self.S.End]      # 终止状态集
        self.Y = SolveMatrix(self, 1) 
    
    # 判断给定状态是否为终止状态
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    # 获得即时奖励，保留此函数可以为将来更复杂的奖励函数做准备
    def get_reward(self, s):
        return self.R[s.value]

    # 根据转移概率前进一步，返回（下一个状态、即时奖励、是否为终止）
    def step(self, curr_s):
        next_s = np.random.choice(self.S, p=self.P[curr_s.value])
        return next_s, self.get_reward(next_s), self.is_end(next_s)

def SolveMatrix(dataModel, gamma):
    # 在对角矩阵上增加一个微小的值来解决奇异矩阵不可求逆的问题
    #I = np.eye(dataModel.N) * (1+1e-7)
    I = np.eye(dataModel.N)
    factor = I - gamma * dataModel.P
    inv_factor = np.linalg.inv(factor)
    vs = np.dot(inv_factor, dataModel.R)
    return vs

if __name__=="__main__":
    dataModel = DataModel()
    v = SolveMatrix(dataModel, 1.0)
    print(v)
    vv = np.around(v,3)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))
    print(RMSE(np.array(a), dataModel.Y))
    print(RMSE(np.array(ground_truth), dataModel.Y))
