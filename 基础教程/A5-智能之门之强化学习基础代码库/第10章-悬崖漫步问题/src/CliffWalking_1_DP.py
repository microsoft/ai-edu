
import gymnasium as gym
import numpy as np
import common.CommonHelper as helper
import tqdm
import common.Algo_MC_OnPolicy_Predict as mc_pred
import common.Algo_DP_PolicyEvaluation as algoPI    


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    #env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=True)
    env.reset(seed=5)
    print(env.unwrapped.P)
    Episodes = 50
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    V = np.zeros(env.observation_space.n)
    alpha = 0.1
    gamma = 0.9

    V_dp, _  = algoPI.calculate_VQ_pi(env, behavior_policy, gamma)
    helper.print_V(V_dp, 1, (4,12), helper.SeperatorLines.middle, "DP")
    V_dp[37:48] = 0
    helper.print_V(V_dp, 0, (4,12), helper.SeperatorLines.middle, "DP")
