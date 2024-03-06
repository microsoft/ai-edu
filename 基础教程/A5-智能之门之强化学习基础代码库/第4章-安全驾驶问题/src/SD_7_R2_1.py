import numpy as np
import common.SimpleModel as Model
import common.Algo_Sampling as Sampler
from enum import Enum


# 状态定义
class States(Enum):
    Start = 0
    A = 1
    B = 2
    C = 3
    D = 4
    Home = 5    # 终止状态

# 状态转移矩阵
P = np.array(
    [  # S    A    B    C    D    Home     
        [0.5, 0.5, 0,   0,   0,   0  ], # S
        [0.5, 0,   0.5, 0,   0,   0  ], # A
        [0,   0.5, 0,   0.5, 0,   0  ], # B
        [0,   0,   0.5, 0,   0.5, 0  ], # C
        [0,   0,   0,   0.5, 0,   0.5], # D
        [0,   0,   0,   0,   0,   1  ]  # Home(End)
])

# 状态奖励值
#     S,   A,  B,  C,  D, H
R = [ 0,  0,  0,  0,  0, 1]

if __name__=="__main__":
    episodes = 1000
    print("奖励值:", R)
    model = Model.SimpleModel(States, P, R2=R)
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))

        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            V[s.value] = Sampler.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        Sampler.print_V(model, V)
