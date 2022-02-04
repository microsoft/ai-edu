import numpy as np
from enum import Enum
import MRP_Algo_Bellman as mab
import Data_FrozenLake as dfl

if __name__=="__main__":
    gamma = 0.9
    vs = mab.run(dfl.States, dfl.Matrix, dfl.Rewards, gamma)
    print(np.round(np.array(vs).reshape(4,4), 2))
