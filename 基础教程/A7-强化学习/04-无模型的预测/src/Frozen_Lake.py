import numpy as np
import Data_FrozenLake as ds
import Algorithm_MRP as algoMRP
import Algorithm_Model_Free_Prediction as algo

def FrozenLake_Matrix(gamma):
    vs = algoMRP.Matrix(ds, gamma)
    print("Matrix")
    print(np.round(np.array(vs).reshape(4,4), 2))

def FrozenLake_Bellman(gamma):
    vs = algoMRP.Bellman(ds, gamma)
    np.set_printoptions(suppress=True)
    print("Bellman")
    print(np.round(np.array(vs).reshape(4,4), 2))


def FrozenLake_MentoCarol(gamma, episodes):
    vs = algo.MonteCarol2(ds, gamma, episodes)
    print("MC")
    print(np.round(np.array(vs).reshape(4,4), 2))


if __name__=="__main__":
    gamma = 1
    FrozenLake_Matrix(gamma)
    FrozenLake_Bellman(gamma)

    episodes = 10000
    FrozenLake_MentoCarol(gamma, episodes)

    alpha = 0.1
    V = np.zeros(16)

    v = algo.MC(V, ds, ds.States.Start, episodes, alpha, gamma)
    print("MC")
    print(np.round(np.array(v).reshape(4,4), 2))

    v = algo.TD(V, ds, ds.States.Start, episodes, alpha, gamma)
    print("TD")
    print(np.round(np.array(v).reshape(4,4), 2))
