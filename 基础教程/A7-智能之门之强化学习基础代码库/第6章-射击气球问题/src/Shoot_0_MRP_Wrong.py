import numpy as np
from enum import Enum
import common.CommonHelper as helper

# 奖励
#    0   1  2  3  4  5  6  7  8  9  10 11 12
R = [-4, 0, 1, 3]

P = np.array(
    [   # 0     1     2     3    
        [0.00, 0.56, 0.38, 0.06], # 0
        [0.00, 0.56, 0.38, 0.06], # 1
        [0.00, 0.50, 0.42, 0.08], # 2
        [0.00, 0.40, 0.50, 0.10], # 3
    ]
)

# 状态
class States(Enum):
    Start = 0
    Miss = 1
    Small = 2
    Grand = 3

class DataModel(object):
    def __init__(self):
        self.P = P                          # 状态转移矩阵
        self.R = R                    # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)       # 状态数量

def SolveMatrix(dataModel: DataModel, gamma: float):
    # 在对角矩阵上增加一个微小的值来解决奇异矩阵不可求逆的问题
    I = np.eye(dataModel.N) * (1+1e-7)
    #I = np.eye(dataModel.N)
    factor = I - gamma * dataModel.P
    inv_factor = np.linalg.inv(factor)
    vs = np.dot(inv_factor, dataModel.R)
    return vs

def check_convergence(dataModel, gamma=1):
    print("迭代100次, 检查状态转移矩阵是否趋近于 0: ")
    P_new = dataModel.P.copy()
    for i in range(100):
        P_new = np.dot(dataModel.P, P_new) * gamma
    print(np.around(P_new, 3))

if __name__=="__main__":
    dataModel = DataModel()
    gammas = [1]
    for gamma in gammas:
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("折扣 =", gamma)
        helper.print_seperator_line(helper.SeperatorLines.middle)
        check_convergence(dataModel, gamma)
        V = SolveMatrix(dataModel, gamma)
        vv = np.around(V,3)
        helper.print_seperator_line(helper.SeperatorLines.middle)
        print("价值函数：")
        for s in dataModel.S:
            print(str.format("{0}:\t{1}", s.name, vv[s.value]))
    