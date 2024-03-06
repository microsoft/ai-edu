
import gymnasium as gym
import numpy as np
import common.CommonHelper as helper
import common.Algo_TD_DQLearning as algoDQL
import common.DrawQpi as drawQ

if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    env.reset(seed=5)
    Episodes = 5000
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    ctrl = algoDQL.TD_DQLearning(env, Episodes, behavior_policy, alpha, gamma, epsilon)
    Q = ctrl.run()
    helper.print_Q(Q, 2, (4,12), helper.SeperatorLines.middle, "Double QLearning - Q")
    helper.print_seperator_line(helper.SeperatorLines.middle, "Double QLearning - Action")
    drawQ.drawQ(Q, (4,12), round=2, goal_state=47, end_state=[37, 38, 39, 40, 41, 42, 43, 44, 45, 46])
    helper.print_Policy(ctrl.behavior_policy, 3, (4,12), helper.SeperatorLines.middle, "Double QLearning - Policy")
    drawQ.drawPolicy(ctrl.behavior_policy, (4,12), round=3, goal_state=47, end_state=[37, 38, 39, 40, 41, 42, 43, 44, 45, 46])
