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
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                for k in range(Q.shape[2]):
                    arg = np.argmax(Q[i,j,k])
                    self.policy[i,j,k] = 0
                    self.policy[i,j,k,arg] = 1

        return self.policy


def get_groud_truth(env, gamma):
    iteration = 100
    _, Q = algoDP.calculate_VQ_star(env, gamma, iteration)
    return Q

if __name__=="__main__":
    gamma = 1
    rough = 1000
    final = 20000
    
    np.set_printoptions(suppress=True)
    env = gym.make('Blackjack-v1', sab=True)
    env.reset(seed=5)
    algo = MC_Greedy(env, (0.5,0.5), gamma, rough, final)
    Q, policy = algo.policy_iteration()
    env.close()
    
    print(policy)

