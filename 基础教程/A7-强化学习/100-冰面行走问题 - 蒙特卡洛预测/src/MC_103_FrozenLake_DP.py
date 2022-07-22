# 用 DP 迭代方法计算随机策略下的 FrozenLake 问题的 V_pi, Q_pi

import numpy as np
import gym
import Algorithm.Algo_PolicyValueFunction as algoPI
import Algorithm.Algo_OptimalValueFunction as algoStar
import common.DrawQpi as drawQ
import common.CommonHelper as helper

if __name__=="__main__":

    slips = [False, True]
    gamma = 1
    iteration = 1000
    for slip in slips:
        helper.print_seperator_line(helper.SeperatorLines.long, str.format("is slippery={0}", slip))
        env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=slip)
        # print(env.P)
        # 随机策略
        policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        V_pi, Q_pi = algoPI.calculate_VQ_pi(env, policy, gamma, iteration)
        helper.print_seperator_line(helper.SeperatorLines.middle, "随机策略下的状态价值函数 V")
        print(np.reshape(np.round(V_pi,3), (4,4)))
        helper.print_seperator_line(helper.SeperatorLines.middle, "随机策略下的动作价值函数 Q")
        print(np.round(Q_pi,3))
        drawQ.draw(Q_pi, (4,4))

        V_star, Q_star = algoStar.calculate_VQ_star(env, gamma, iteration)
        helper.print_seperator_line(helper.SeperatorLines.middle, "最优策略下的状态价值函数 V")
        print(np.reshape(np.round(V_star,3), (4,4)))
        helper.print_seperator_line(helper.SeperatorLines.middle, "最优策略下的动作价值函数 Q")
        Q_star = np.round(Q_star,3)
        print(Q_star)
        drawQ.draw(Q_star, (4,4))
        env.close()
