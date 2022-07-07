import gym
import Algorithm.Algo_MonteCarlo_MDP as algo
import common.DrawQpi as drawQ
import numpy as np
import common.CommonHelper as helper

if __name__=="__main__":
    gamma = 1
    episodes = 50000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=True)
    start_state, info = env.reset(seed=5, return_info=True)
    Q1 = algo.MC_EveryVisit_Q(env, start_state, episodes, gamma)
    Q2 = algo.MC_EveryVisit_Q(env, start_state, episodes, gamma)
    Q3 = algo.MC_EveryVisit_Q(env, start_state, episodes, gamma)
    '''
    V1 = algo.MC_EveryVisit_V(env, start_state, episodes, gamma)
    V2 = algo.MC_EveryVisit_V(env, start_state, episodes, gamma)
    V3 = algo.MC_EveryVisit_V(env, start_state, episodes, gamma)
    env.close()
    print(np.reshape(V1, (4,4)))
    print(np.reshape(V2, (4,4)))
    print(np.reshape(V3, (4,4)))
    V = (V1 + V2 + V3)/3
    print(helper.RMSE(V1, V))
    print(helper.RMSE(V2, V))
    print(helper.RMSE(V3, V))
    '''
    Q = (Q1 + Q2 + Q3)/3
    drawQ.draw(Q1, (4,4))
    drawQ.draw(Q2, (4,4))
    drawQ.draw(Q3, (4,4))

    print(Q)
    drawQ.draw(Q, (4,4))
    
