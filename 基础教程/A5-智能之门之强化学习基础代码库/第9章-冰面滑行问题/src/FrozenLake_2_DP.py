# 用 DP 迭代方法计算随机策略下的 FrozenLake 问题的 V_pi, Q_pi

import numpy as np
import gymnasium as gym
import common.Algo_DP_PolicyEvaluation as algoPI
import common.Algo_DP_ValueIteration as algoStar
import common.DrawQpi as drawQ
import common.CommonHelper as helper


if __name__=="__main__":

    slips = [False, True]
    for slip in slips:
        helper.print_seperator_line(helper.SeperatorLines.long, str.format("is slippery={0}", slip))
        env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=slip)
        helper.print_seperator_line(helper.SeperatorLines.middle, "状态转移概率矩阵")
        print(env.unwrapped.P)
        helper.print_seperator_line(helper.SeperatorLines.middle, "随机策略价值估算")
        # 随机策略
        gamma = 1.0
        policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        V_pi, Q_pi = algoPI.calculate_VQ_pi(env, policy, gamma)
        helper.print_V(V_pi, 3, (4,4), helper.SeperatorLines.short, "随机策略下的状态价值函数 V")
        helper.print_Q(Q_pi, 3, (4,4), helper.SeperatorLines.short, "随机策略下的动作价值函数 Q")
        drawQ.drawQ(Q_pi, (4,4), round=3, goal_state=15)

        helper.print_seperator_line(helper.SeperatorLines.middle, "最优策略价值估算")
        # 最优策略
        gamma = 0.9
        V_star, Q_star = algoStar.calculate_VQ_star(env, gamma)
        helper.print_V(V_star, 3, (4,4), helper.SeperatorLines.short, "最优策略下的状态价值函数 V")
        helper.print_Q(Q_star, 3, (4,4), helper.SeperatorLines.short, "最优策略下的动作价值函数 Q")
        drawQ.drawQ(Q_star, (4,4), round=3, goal_state=15)
        env.close()
    

