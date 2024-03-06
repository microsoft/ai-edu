import numpy as np
from enum import Enum
import common.Algo_Sampling as Sampler
import common.SimpleModel as Model


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

#   Start, A, B, C, D, H, End
R = [0,   0, 0, 0, 0, 1, 0]


if __name__=="__main__":
    episodes = 1000
    print("奖励值:", R)
    model = Model.SimpleModel(States, P, R2=R)
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))
        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            print("起始状态:", s)
            V[s.value] = Sampler.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        Sampler.print_V(model, V)
