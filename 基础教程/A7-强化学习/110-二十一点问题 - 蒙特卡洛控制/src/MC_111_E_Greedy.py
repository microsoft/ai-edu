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


class MC_E_Greedy(base.Policy_Iteration):
    def __init__(self, env, policy, gamma:float, episodes:int, final:int, epsilon:float):
        super().__init__(env, policy, gamma, episodes, final)
        self.epsilon = epsilon
        self.other_p = self.epsilon / self.nA
        self.best_p = 1 - self.epsilon + self.epsilon / self.nA
    
    def policy_improvement(self, Q):
        print(np.sum(Q))

        for s in range(self.nS):
            if s in end_states:
                self.policy[s] = 0
            else:
                max_A = np.max(Q[s])
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
    rough = 300
    final = 10000
    epsilon = 0.05
    
    np.set_printoptions(suppress=True)
    #desc = ["SFHF","FFFF","HFHF","FFFF","FFFG"]
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=True)
    end_states = [5, 7, 11, 12, 15]
    helper.print_seperator_line(helper.SeperatorLines.long, "动态规划法")
    Q_real = get_groud_truth(env, gamma)
    helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
    print(np.round(Q_real,3))
    drawQ.draw(Q_real,(4,4))
    

    policy = helper.create_policy(env.observation_space.n, env.action_space.n, (0.25,0.25,0.25,0.25))
    env.reset(seed=5)
    algo = MC_E_Greedy(env, policy, gamma, rough, final, epsilon)
    Q, policy = algo.policy_iteration()
    env.close()

    helper.print_seperator_line(helper.SeperatorLines.long, "蒙特卡洛法 - E-Greedy")
    helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
    print(np.round(Q,3))
    drawQ.draw(Q,(4,4))
    print(policy)

