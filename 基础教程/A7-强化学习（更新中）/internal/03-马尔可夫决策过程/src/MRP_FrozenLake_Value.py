import numpy as np
import Algorithm_MRP as algoM
import Data_FrozenLake as dfl


def FrozenLake_Matrix(gamma):
    vs = algoM.Matrix(dfl, gamma)
    print("Matrix:")
    print(np.round(np.array(vs).reshape(4,4), 3))

def FrozenLake_Bellman(gamma):
    vs = algoM.Bellman(dfl, gamma)
    np.set_printoptions(suppress=True)
    print("Bellma:")
    print(np.round(np.array(vs).reshape(4,4), 3))


if __name__=="__main__":
    gamma = 1
    FrozenLake_Matrix(gamma)
    FrozenLake_Bellman(gamma)
