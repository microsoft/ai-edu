import numpy as np
import Algo_Sampling as algo
from RandomWalker_2_DataModel import *

# 过程奖励值
R1 = {
    States.Start: {States.Start: 0,    States.A:0},     # S->S:0, S->A:0
    States.A:     {States.Start:0,     States.B:0},     # A->S:0, A->B:0
    States.B:     {States.A:0,         States.C:0},     # B->A:0, B->C:0
    States.C:     {States.B:0,         States.D:0},     # C->B:0, C->D:0
    States.D:     {States.C:0,         States.Home:1},  # D->C:0, D->H:1
    States.Home:  {States.Home:0},                      # H->H:0
}

if __name__=="__main__":
    episodes = 1000
    print("奖励值:", R1)
    model = DataModel(R1)
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))
        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            V[s.value] = algo.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        algo.print_V(model, V)
