import numpy as np
from enum import Enum
import copy
import Shoot_2_DataModel as dataModel
import Algo_OptimalValueFunction as algo

if __name__=="__main__":
    env = dataModel.Env(None)
    gamma = 1
    max_iteration = 100
    V_star, Q_star = algo.calculate_Vstar(env, gamma, max_iteration)
    print(np.round(V_star,5))
    policy = algo.get_policy(env, V_star, gamma)
    print(policy)
    print(Q_star)



