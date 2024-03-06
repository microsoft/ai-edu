
import gymnasium as gym
import numpy as np
import common.CommonHelper as helper
import common.Algo_DP_PolicyEvaluation as algoPI    
import common.Algo_TD_TD0 as algoTD0


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    env.reset(seed=5)
    Episodes = 50
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    alpha = 0.1
    gamma = 0.9

    V_dp, _ = algoPI.calculate_VQ_pi(env, behavior_policy, gamma)
    V_dp[37:48] = 0
    helper.print_V(V_dp, 0, (4,12), helper.SeperatorLines.middle, "DP")

    pred = algoTD0.TD_TD0(env, Episodes, behavior_policy, alpha, gamma)
    V = pred.run()
    helper.print_V(V, 0, (4,12), helper.SeperatorLines.middle, "TD")
    print(helper.Norm2Err(V, V_dp))

