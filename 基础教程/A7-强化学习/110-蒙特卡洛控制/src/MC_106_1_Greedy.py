from cProfile import label
from email import policy
import numpy as np
import gym
import Algorithm.Algo_MC_Policy_Iteration as algoMC
import Algorithm.Algo_OptimalValueFunction as algoDP
import common.DrawQpi as drawQ
import common.CommonHelper as helper
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False


class MC_Greedy(algoMC.Policy_Iteration):
    def policy_improvement(self, Q):
        for s in range(self.nS):
            arg = np.argmax(Q[s])
            self.policy[s] = 0
            self.policy[s, arg] = 1

        return self.policy


def get_groud_truth(env, gamma):
    iteration = 100
    _, Q = algoDP.calculate_VQ_star(env, gamma, iteration)
    return Q


if __name__=="__main__":
    gamma = 0.9
    episodes = 1000
    final = 2000
    
    np.set_printoptions(suppress=True)
    env = gym.make("FrozenLake-v1", desc=None, map_name = "8x8", is_slippery=False)
    Q_real = get_groud_truth(env, gamma)
    print(np.round(Q_real,3))
    drawQ.draw(Q_real,(8,8))

    policy = helper.create_policy(env, (0.25,0.25,0.25,0.25))
    env.reset(seed=5)
    algo = MC_Greedy(env, policy, gamma, episodes, final)
    Q, policy = algo.policy_iteration()
    env.close()
    
    print("------ 最优动作价值函数 -----")
    print(np.round(Q,3))
    drawQ.draw(policy,(8,8))
    print(policy)

