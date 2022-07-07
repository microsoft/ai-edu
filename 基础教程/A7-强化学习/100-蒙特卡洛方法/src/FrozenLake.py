import gym
import Algorithm.Algo_MonteCarlo_MDP as algo
import common.DrawQpi as drawQ
import numpy as np

if __name__=="__main__":
    gamma = 1
    episodes = 10000
    #env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=True)
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4")
    start_state, info = env.reset(seed=5, return_info=True)
    Q = algo.MC_EveryVisit_Q(env, start_state, episodes, gamma)
    #V1 = algo.MC_EveryVisit_V(env, start_state, episodes, gamma)
    #V2 = algo.MC_EveryVisit_V(env, start_state, episodes, gamma)
    env.close()
    #print(np.reshape(V1, (4,4)))
    #print(np.reshape(V2, (4,4)))
    print(Q)
    drawQ.draw(Q, (8,8))
    
