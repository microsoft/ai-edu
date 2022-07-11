import numpy as np
import gym
import Algorithm.Algo_MonteCarlo_MDP as algoMC
import Algorithm.Algo_OptimalValueFunction as algoDP
import common.DrawQpi as drawQ
import common.CommonHelper as helper

def get_groud_truth(env, gamma):
    iteration = 100
    V, Q = algoDP.calculate_VQ_star(env, gamma, iteration)
    return V, Q

if __name__=="__main__":
    gamma = 1
    episodes = 10000
    env = gym.make("FrozenLake-v1", map_name = "4x4", is_slippery=True)
    # 初始随机策略
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    nA = env.action_space.n
    nS = env.observation_space.n
    policy = np.ones(shape=(nS, nA)) / nA   # 随机策略，每个状态上的每个动作都有0.25的备选概率
    start_state, info = env.reset(seed=5, return_info=True)
    Q = algoMC.MC_EveryVisit_Q_GLIE(env, start_state, episodes, gamma, policy)
    print(np.round(Q,3))
    drawQ.draw(Q,(4,4))

    R = helper.test_policy(env, policy, episodes=100)
    print("Total Reward =", R)

    env.close()