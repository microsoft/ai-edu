
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
#Rewards = [-1, -2, -2, -2, 10, 1, 0]
Rewards = [-2, 0.5, 1, 1.5, 10, -2, 0]

# 状态转移概率
P = np.array(
    [   #Game Cl1  Cl2  Cl3  Pass Rest End
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
    ]
)

ground_truth = [-22.54, -12.54, 1.46, 4.32, 10.00, 0.80, 0.00]

class DataModel(object):
    def __init__(self):
        self.P = P                          # 状态转移矩阵
        self.R = Rewards                    # 奖励
        self.S = States                     # 状态集
        self.num_states = len(self.S)       # 状态数量
        self.end_states = [self.S.End]      # 终止状态集
        self.Y = ground_truth       
    
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

def Matrix(dataModel, gamma):
    num_state = dataModel.P.shape[0]
    I = np.eye(dataModel.num_states) * (1+1e-7)
    tmp1 = I - gamma * dataModel.P
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, dataModel.R)
    return vs

if __name__=="__main__":
    dataModel = DataModel()
    v = Matrix(dataModel, 1.0)
    print(np.around(v,1))
