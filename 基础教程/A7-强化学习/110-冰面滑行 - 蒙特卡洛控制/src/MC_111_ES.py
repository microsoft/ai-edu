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
            if s in end_states:
                self.policy[s] = 0
            else:
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
    rough = 100
    final = 2000
    
    np.set_printoptions(suppress=True)
    #desc = ["SFHF","FFFF","HFHF","FFFF","FFFG"]
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    end_states = [5, 7, 11, 12, 15]
    Q_real = get_groud_truth(env, gamma)
    print(np.round(Q_real,3))
    #drawQ.draw(Q_real,(4,4))
    

    policy = helper.create_policy(env.observation_space.n, env.action_space.n, (0.25,0.25,0.25,0.25))
    env.reset(seed=5)
    algo = MC_Greedy(env, policy, gamma, rough, final)
    Q, policy = algo.policy_iteration()
    env.close()
    
    print("------ 最优动作价值函数 -----")
    print(np.round(Q,3))
    drawQ.draw(policy,(4,4))
    print(policy)

