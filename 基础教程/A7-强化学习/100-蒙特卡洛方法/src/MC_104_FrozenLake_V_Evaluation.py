import numpy as np
import gym
import Algorithm.Algo_MonteCarlo_MDP as algoMC
import Algorithm.Algo_PolicyValueFunction as algoDP
import common.DrawQpi as drawQ
import common.CommonHelper as helper

def get_groud_truth(env, policy, gamma):
    iteration = 100
    V, Q = algoDP.calculate_VQ_pi(env, policy, gamma, iteration)
    return V, Q

if __name__=="__main__":
    gamma = 1
    episodes = 1000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    V_real, Q_real = get_groud_truth(env, policy, gamma)


    nA = env.action_space.n
    nS = env.observation_space.n
    policy = np.ones(shape=(nS, nA)) / nA   # 随机策略，每个状态上的每个动作都有0.25的备选概率
    start_state, info = env.reset(seed=5, return_info=True)
    V = algoMC.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    print(np.reshape(np.round(V,3),(4,4)))

    env.close()
    print(helper.RMSE(V, V_real))