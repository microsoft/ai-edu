
import numpy as np
from enum import Enum
import common_helper as helper

# 状态
class States(Enum):
    Bug = 0
    Coding = 1
    Test = 2
    Review = 3
    Refactor = 4
    Merge = 5
    End = 6

# 奖励向量 缺陷 编码 测试 审查 重构 合并 结束
Rewards = [-3, 0,   +1, +3,  +2, -1,  0]

# 状态转移概率
P = np.array(
    [   # B   C    T    R    F    M    E    
        [0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],    # Bug 
        [0.6, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],    # Coding
        [0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0],    # Test (CI)
        [0.0, 0.1, 0.0, 0.0, 0.2, 0.7, 0.0],    # Review
        [0.0, 0.2, 0.5, 0.3, 0.0, 0.0, 0.0],    # reFactor
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],    # Merge
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # End
    ]
)

class DataModel(object):
    def __init__(self):
        self.P = P                          # 状态转移矩阵
        self.R = Rewards                    # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)       # 状态数量
        self.end_states = [self.S.End]      # 终止状态集

def solve_matrix(dataModel, gamma):
    # 在对角矩阵上增加一个微小的值来解决奇异矩阵不可求逆的问题
    # I = np.eye(dataModel.N) * (1+1e-7)
    I = np.eye(dataModel.N)
    factor = I - gamma * dataModel.P
    inv_factor = np.linalg.inv(factor)
    vs = np.dot(inv_factor, dataModel.R)
    return vs

if __name__=="__main__":
    dataModel = DataModel()
    V = solve_matrix(dataModel, 1.0)
    helper.print_V(dataModel, V)
