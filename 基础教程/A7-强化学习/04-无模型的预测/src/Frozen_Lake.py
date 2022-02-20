import numpy as np
import Data_FrozenLake as ds
import Algorithm_MRP as algoMRP
import Algorithm_Model_Free_Prediction as algo

def FrozenLake_Matrix(gamma):
    vs = algoMRP.Matrix(ds, gamma)
    print("Matrix")
    print(np.round(np.array(vs).reshape(4,4), 2))
    return vs

def FrozenLake_Bellman(gamma):
    vs = algoMRP.Bellman(ds, gamma)
    np.set_printoptions(suppress=True)
    print("Bellman")
    print(np.round(np.array(vs).reshape(4,4), 2))

def FrozenLake_MentoCarol(gamma, episodes):
    vs = algo.MonteCarol(ds.Data_Frozen_Lake(), gamma, episodes)
    print("MC - each state")
    print(np.round(np.array(vs).reshape(4,4), 2))


def RSME(a, b):
    err = np.sqrt(np.sum(np.square(a - b))/V.shape[0])
    print("RSME=",err)

def set_end_state_value(v):
    v[2] = -1
    v[8] = -1
    v[10] = -1
    v[15] = 5
    return v

if __name__=="__main__":
    gamma = 0.9
    ground_truth = FrozenLake_Matrix(gamma)
    FrozenLake_Bellman(gamma)

    episodes = 10000
    #FrozenLake_MentoCarol(gamma, episodes)

    alpha = 0.02
    V = np.zeros(16)

    v = algo.MC(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
    print("\nMC - all state - 1")
    print(np.round(np.array(v).reshape(4,4), 2))
    set_end_state_value(v)
    RSME(v, ground_truth)

    V = np.zeros(16)
    v = algo.MC2(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
    print("\nMC - all state - 2")
    print(np.round(np.array(v).reshape(4,4), 2))
    v = set_end_state_value(v)
    RSME(v, ground_truth)

    V = np.zeros(16)
    v = algo.TD(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
    print("\nTD")
    print(np.round(np.array(v).reshape(4,4), 2))
    v = set_end_state_value(v)
    RSME(v, ground_truth)
