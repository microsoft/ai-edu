import gym
import Algorithm.Algo_MonteCarlo_MDP as algo
import common.DrawQpi as drawQ
import numpy as np
import common.CommonHelper as helper

ground_truth = np.array([
    0.014, 0.012, 0.021, 0.010,
    0.016, 0, 0.041, 0,
    0.035, 0.088, 0.142, 0,
    0, 0.176, 0.439, 0
])

if __name__=="__main__":
    gamma = 1
    episodes = 50000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)

    nA = env.action_space.n
    nS = env.observation_space.n
    policy = np.ones(shape=(nS, nA)) / nA   # 随机策略，每个状态上的每个动作都有0.25的备选概率
    start_state, info = env.reset(seed=5, return_info=True)
    V = algo.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    print(np.reshape(np.round(V,3),(4,4)))

    Q = algo.MC_EveryVisit_Q_Policy(env, start_state, episodes, gamma, policy)
    print(np.round(Q,3))
    drawQ.draw(Q,(4,4))


    env.close()
    exit(0)


    episodes = 50000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=True)

    nA = env.action_space.n
    nS = env.observation_space.n
    policy = np.ones(shape=(nS, nA)) / nA   # 随机策略，每个状态上的每个动作都有0.25的备选概率
    start_state, info = env.reset(seed=5, return_info=True)
    V1 = algo.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    print(np.reshape(np.round(V1,3),(4,4)))
    env.close()

    print(helper.RMSE(V1, ground_truth))

    episodes = 15000
    V1 = algo.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    V2 = algo.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    V3 = algo.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    print(np.reshape(V1, (4,4)))
    print(helper.RMSE(V1, ground_truth))
    print(np.reshape(V2, (4,4)))
    print(helper.RMSE(V2, ground_truth))
    print(np.reshape(V3, (4,4)))
    print(helper.RMSE(V3, ground_truth))
    V = (V1 + V2 + V3)/3
    print(np.reshape(V, (4,4)))
    print(helper.RMSE(ground_truth, V))


    env.close()
    exit(0)

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
    
