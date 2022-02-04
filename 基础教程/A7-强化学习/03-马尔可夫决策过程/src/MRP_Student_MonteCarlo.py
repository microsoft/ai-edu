import math
import numpy as np
import tqdm
import multiprocessing as mp

import Data_Student as ds
import MRP_Algo_MonteCarlo as algoM

if __name__=="__main__":
    gamma = 0.9
    episodes = 10000
    end_states = [ds.States.Sleep]
    algoM.run(ds.Rewards, ds.Matrix, ds.States, end_states, gamma, episodes)
