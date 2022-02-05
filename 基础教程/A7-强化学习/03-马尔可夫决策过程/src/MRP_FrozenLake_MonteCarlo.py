import math
import numpy as np
import tqdm
import multiprocessing as mp

import Data_FrozenLake as dfl
import MRP_Algo_MonteCarlo as algoM

if __name__=="__main__":
    gamma = 0.9
    episodes = 50000
    end_states = [dfl.States.End, dfl.States.Hole11, dfl.States.Hole12]
    vs = algoM.run(dfl.Rewards, dfl.Matrix, dfl.States, end_states, gamma, episodes)
    print(np.round(np.array(vs).reshape(4,4), 2))