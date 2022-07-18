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


class MC_GLIE(algoMC.Policy_Iteration):
    def __init__(self, env, policy, episodes, gamma, epsilon):
        super().__init__(env, policy, episodes, gamma)
        self.epsilon = epsilon
    
    def initialize(self):
        super().initialize()
        epsilon = 1
        self.other_p = epsilon / self.nA
        self.best_p = 1 - epsilon + epsilon/self.nA

    def policy_improvement(self, Q):
        epsilon = 1 / (math.log(k+1)+1)
        other_p = epsilon / self.nA
        best_p = 1 - epsilon + epsilon/self.nA

        # 更新策略        
        for s in range(self.nS):
            max_A = np.max(Q[s])
            if max_A == 0:
                self.policy[s] = 0
            else:
                argmax_A = np.where(Q[s] == max_A)[0]
                A = np.random.choice(argmax_A)
                self.policy[s] = self.other_p
                self.policy[s,A] = self.best_p

        return self.policy

def get_groud_truth(env, gamma):
    iteration = 100
    _, Q = algoDP.calculate_VQ_star(env, gamma, iteration)
    return Q


if __name__=="__main__":
    gamma = 0.9
    episodes = 1000
    
    np.set_printoptions(suppress=True)
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    Q_real = get_groud_truth(env, gamma)
    print(np.round(Q_real,3))
    drawQ.draw(Q_real,(4,4))

    policy = helper.create_policy(env, (0.25,0.25,0.25,0.25))
    env.reset(seed=5)
    algo = MC_GLIE(env, policy, episodes, gamma)
    Q, policy = algo.policy_iteration()
    env.close()
    
    print("------ 最优动作价值函数 -----")
    print(np.round(Q,3))
    drawQ.draw(Q,(4,4))
    print(policy)

