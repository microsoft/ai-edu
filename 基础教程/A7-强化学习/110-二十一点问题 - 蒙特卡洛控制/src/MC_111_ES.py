from email import policy
import numpy as np
import gym
import Algorithm.Algo_OptimalValueFunction as algoDP
import Algorithm.Base_MC_Policy_Iteration as base
import common.DrawQpi as drawQ
import common.CommonHelper as helper
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False


class MC_Greedy(base.Policy_Iteration):
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
    gamma = 1
    rough = 100
    final = 2000
    
    np.set_printoptions(suppress=True)
    env = gym.make('Blackjack-v1', sab=True)
    env.reset(seed=5)
    algo = MC_Greedy(env, policy, gamma, rough, final)
    Q, policy = algo.policy_iteration()
    env.close()
    
    print(policy)

