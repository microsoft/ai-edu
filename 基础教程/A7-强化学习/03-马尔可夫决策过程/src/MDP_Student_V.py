from enum import Enum
import imp
import numpy as np
import math

import MDP_Algo_V as algoV
import Data_Students2 as ds2

if __name__=="__main__":
    gamma = 1
    v = algoV.V_pi(ds2.States, ds2.Pi_sa, ds2.P_as, ds2.Rewards, gamma)
    for start_state in ds2.States:
        print(start_state, "= {:.1f}".format(v[start_state.value]))
