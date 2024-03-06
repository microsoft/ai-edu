import numpy as np
import common.Algo_Sampling as Sampler
import common.SimpleModel as Model
from enum import Enum


# 状态定义
class States(Enum):
    End = 0  # 在酒馆左侧增加一个终止状态
    Start = 1
    A = 2
    B = 3
    C = 4
    D = 5
    Home = 6

# 状态转移矩阵
P = np.array(
    [  # E    S    A    B    C    D    H       
        [1,   0,   0,   0,   0,   0,   0], # End
        [0.5, 0,   0.5, 0,   0,   0,   0], # S
        [0,   0.5, 0,   0.5, 0,   0,   0], # A
        [0,   0,   0.5, 0,   0.5, 0,   0], # B
        [0,   0,   0,   0.5, 0,   0.5, 0], # C
        [0,   0,   0,   0,   0.5, 0,   0.5], # D
        [0,   0,   0,   0,   0,   0,   1]  # Home
])

# 过程奖励值
R = {
    States.End: {States.End:0},
    States.Start: {States.End:0,    States.A:0},
    States.A: {States.Start:0,    States.B:0},
    States.B: {States.A:0,        States.C:0},
    States.C: {States.B:0,        States.D:0},
    States.D: {States.C:0,        States.Home:1},
    States.Home: {States.Home:0},
}


if __name__=="__main__":
    episodes = 5000
    print("奖励值:", R)
    model = Model.SimpleModel(States, P, R1=R, end_states=[States.End, States.Home])
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))
        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            print("起始状态:", s.name)
            V[s.value] = Sampler.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        Sampler.print_V(model, V)
