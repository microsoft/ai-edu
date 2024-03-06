import numpy as np
import common.Algo_Sampling as Sampler
import common.SimpleModel as Model
from enum import Enum


# 状态定义
class States(Enum):
    Start = 0
    A = 1
    B = 2
    C = 3
    D = 4
    Home = 5

# 状态转移矩阵
P = np.array(
    [  # S    A    B    C    D    H       
        [0.5, 0.5, 0,   0,   0,   0], # S
        [0.5, 0,   0.5, 0,   0,   0], # A
        [0,   0.5, 0,   0.5, 0,   0], # B
        [0,   0,   0.5, 0,   0.5, 0], # C
        [0,   0,   0,   0.5, 0,   0.5], # D
        [0,   0,   0,   0,   0,   1]  # Home
])

# 过程奖励值
R = {
    States.Start: {
        States.Start: 0,    States.A:0},     # S->S:0, S->A:0
    States.A: {
        States.Start:0,     States.B:0},     # A->S:0, A->B:0
    States.B: {
        States.A:0,         States.C:0},     # B->A:0, B->C:0
    States.C: {
        States.B:0,         States.D:0},     # C->B:0, C->D:0
    States.D: {
        States.C:0,         States.Home:1},  # D->C:0, D->H:1
    States.Home: {
        States.Home:0},                      # H->H:0
}

if __name__=="__main__":
    episodes = 1000
    print("奖励值:", R)
    model = Model.SimpleModel(States, P, R1=R)
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))
        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            V[s.value] = Sampler.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        Sampler.print_V(model, V)
