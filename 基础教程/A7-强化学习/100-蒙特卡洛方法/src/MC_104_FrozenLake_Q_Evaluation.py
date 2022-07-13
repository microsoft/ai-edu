from email import policy
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

def create_policy(env, args):
    left = args[0]
    down = args[1]
    right = args[2]
    up = args[3]
    assert(left+down+right+up==1)
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    policy[:, 0] = left
    policy[:, 1] = down
    policy[:, 2] = right
    policy[:, 3] = up
    return policy

if __name__=="__main__":
    gamma = 1
    episodes = 10000
    policies = [
        (0.25, 0.25, 0.25, 0.25), (0.40, 0.10, 0.10, 0.40)
    ]
    np.set_printoptions(suppress=True)
    for policy_data in policies:
        env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
        policy = create_policy(env, policy_data)
        V_real, Q_real = get_groud_truth(env, policy, gamma)
        nA = env.action_space.n
        nS = env.observation_space.n
        start_state, info = env.reset(seed=5, return_info=True)
        Q = algoMC.MC_EveryVisit_Q_Policy(env, start_state, episodes, gamma, policy)
        Q4 = np.round(Q,4)
        print(Q)
        drawQ.draw(Q,(4,4))
        env.close()
        print(helper.RMSE(Q, Q_real))
