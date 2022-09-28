import numpy as np
import RandomWalker_1_DataModel as data
import Algo_Sampling as algo

# 状态奖励值
#     S,   A,  B,  C,  D, H
R1 = [ 0,  0,  0,  0,  0, 1]
#R2 = [-1, -1, -1, -1, -1, 0]

if __name__=="__main__":
    episodes = 1000
    print("奖励值:", R1)
    model = data.DataModel(R1)
    gammas = [1, 0.9]
    for gamma in gammas:
        V = np.zeros((model.nS))

        for s in model.S:    # 遍历状态集中的每个状态作为起始状态
            V[s.value] = algo.Sampling(model, s, episodes, gamma)

        print("gamma =", gamma)
        algo.print_V(model, V)
