import numpy as np
import tqdm
import math
import Algo_Sampling as algo
from RandomWalker_2_DataModel import *

# 过程奖励值
R1 = {
    States.Start: {States.Start: 0,    States.A:0},
    States.A:     {States.Start:0,     States.B:0},
    States.B:     {States.A:0,         States.C:0},
    States.C:     {States.B:0,         States.D:0},
    States.D:     {States.C:0,         States.Home:1},
    States.Home:  {States.Home:0},
}

R2 = {
    States.Start: {States.Start:-1,    States.A:-1},
    States.A:     {States.Start:-1,    States.B:-1},
    States.B:     {States.A:-1,        States.C:-1},
    States.C:     {States.B:-1,        States.D:-1},
    States.D:     {States.C:-1,        States.Home:0},
    States.Home:  {States.Home:0},
}

def print_V(V):
    print(V)
    vv = np.around(V,3)
    for s in dataModel.S:
        print(str.format("\t\t{0}:\t{1}", s.name, vv[s.value]))

R3 = [
    0,  # S
    0,  # A
    0,  # B
    0,  # C
    0.5,   # D
    0,                  # Home
]
R4 = [
    (-1)*0.5+(-1)*0.5,  # S
    (-1)*0.5+(-1)*0.5,  # A
    (-1)*0.5+(-1)*0.5,  # B
    (-1)*0.5+(-1)*0.5,  # C
    (-1)*0.5+(0)*0.5,  # D
    0*1.0,                 # Home
]

if __name__=="__main__":
    episodes = 1000
    Rs = [R1, R2]
    for R in Rs:
        print("奖励值:", R)
        dataModel = DataModel(R)
        gammas = [1,0.9]
        for gamma in gammas:
            print("gamma =", gamma)
            for start_state in dataModel.S:    # 遍历状态集中的每个状态作为起始状态
                v_s = algo.Sampling(dataModel, start_state, episodes, gamma)
                print(start_state, v_s)
