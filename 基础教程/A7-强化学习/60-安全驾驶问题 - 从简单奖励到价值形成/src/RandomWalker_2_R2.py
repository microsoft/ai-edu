import numpy as np
import Algo_Sampling as algo
from RandomWalker_2_DataModel import *

# 过程奖励值
R2 = {
    States.Start: {States.Start:-1,    States.A:-1},
    States.A:     {States.Start:-1,    States.B:-1},
    States.B:     {States.A:-1,        States.C:-1},
    States.C:     {States.B:-1,        States.D:-1},
    States.D:     {States.C:-1,        States.Home:0},
    States.Home:  {States.Home:0},
}

if __name__=="__main__":
    episodes = 1000
    print("奖励值:", R2)
    model = DataModel(R2)
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))
        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            V[s.value] = algo.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        algo.print_V(model, V)
