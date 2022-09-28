import numpy as np
import RandomWalker_1_DataModel as data
import Algo_Sampling as algo

R3 = [
    0,  # S
    0,  # A
    0,  # B
    0,  # C
    0.5,   # D
    0,                  # Home
]

R4 = [
    0.5*(-1)+0.5*(-1),  # S->S,A = -1
    0.5*(-1)+0.5*(-1),  # A->S,B = -1
    0.5*(-1)+0.5*(-1),  # B->A,C = -1
    0.5*(-1)+0.5*(-1),  # C->B,D = -1
    0.5*(-1)+0.5*0,     # D->C,H = -0.5
    1.0*0               # H->H   = 0
]

if __name__=="__main__":
    episodes = 1000
    Rs = [R3, R4]
    for R in Rs:
        print("奖励值:", R)
        dataModel = data.DataModel(R)
        gammas = [1,0.9]
        for gamma in gammas:
            V = np.zeros((dataModel.nS))
            print("gamma =", gamma)
            for s in dataModel.S:    # 遍历状态集中的每个状态作为起始状态
                V[s.value] = algo.Sampling(dataModel, s, episodes, gamma)
            algo.print_V(dataModel, V)
