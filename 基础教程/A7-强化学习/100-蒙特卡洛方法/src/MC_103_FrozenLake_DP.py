# 用 DP 迭代方法计算随机策略下的 FrozenLake 问题的 V_pi, Q_pi

import numpy as np
import gym
import Algorithm.Algo_PolicyValueFunction as algoPI
import Algorithm.Algo_OptimalValueFunction as algoStar
import common.DrawQpi as drawQ

if __name__=="__main__":

    slips = [False, True]
    gamma = 1
    iteration = 100
    for slip in slips:
        env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=slip)
        # print(env.P)
        # 随机策略
        policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        V_pi, Q_pi = algoPI.calculate_VQ_pi(env, policy, gamma, iteration)
        V_star, Q_star = algoStar.calculate_VQ_star(env, gamma, iteration)
        print("随机策略下的状态价值函数, is_slippery =", slip)
        print(np.reshape(np.round(V_pi,3), (4,4)))
        print("随机策略下的动作价值函数")
        print(np.round(Q_pi,3))
        drawQ.draw(Q_pi, (4,4))

        print("最优状态价值函数, is_slippery =", slip)
        print(np.reshape(np.round(V_star,3), (4,4)))
        print("最优动作价值函数")
        print(np.round(Q_star,3))
        drawQ.draw(Q_star, (4,4))
        env.close()
