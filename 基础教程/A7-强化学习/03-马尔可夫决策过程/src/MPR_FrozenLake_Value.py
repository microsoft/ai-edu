import numpy as np
import Algorithm_MPR as algoM
import Data_FrozenLake as dfl


def FrozenLake_MentoCarol(gamma):
    episodes = 20000
    end_states = [dfl.States.Hole2, dfl.States.Hole8, dfl.States.Hole10, dfl.States.Goal15]
    vs = algoM.MonteCarol(dfl.Rewards, dfl.Matrix, dfl.States, end_states, gamma, episodes)
    print(np.round(np.array(vs).reshape(4,4), 2))

def FrozenLake_Matrix(gamma):
    vs = algoM.Matrix(dfl, gamma)
    print(np.round(np.array(vs).reshape(4,4), 2))

def FrozenLake_Bellman(gamma):
    vs = algoM.Bellman(dfl.States, dfl.Matrix, dfl.Rewards, gamma)
    np.set_printoptions(suppress=True)
    print(np.round(np.array(vs).reshape(4,4), 2))


if __name__=="__main__":
    gamma = 1
    print(gamma)
    #FrozenLake_MentoCarol(gamma)
    FrozenLake_Matrix(gamma)
    FrozenLake_Bellman(gamma)
