import numpy as np
import gymnasium as gym
import common.CommonHelper as helper
import common.Algo_DP_PolicyEvaluation as algo_DP
import common.Algo_MC_OffPolicy_Predict as algo_MC_OffPolicy_Predict
    

if __name__=="__main__":
    env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=True)
    env.reset(seed=5)
    gamma = 0.9
    episodes = 100000
    # 读取上一小节中的最佳策略作为目标策略
    soft_policy = np.load("best_policy.npy")
    helper.print_Policy(soft_policy, 4, (4,4), helper.SeperatorLines.middle, "9.6节的最佳参数")
    target_policy = helper.Soft2Hard(soft_policy)
    # 行动策略初始化为随机策略
    behavior_policy = helper.create_policy(
        env.observation_space.n, env.action_space.n, (0.25, 0.25, 0.25, 0.25))
    
    V_dp, Q_dp = algo_DP.calculate_VQ_pi(env, target_policy, gamma)
    pred = algo_MC_OffPolicy_Predict.MC_OffPolicy_Predict_VQ(
        env, episodes, gamma, behavior_policy, target_policy)
    V, Q = pred.run()

    print("V 误差=",helper.RMSE(V, V_dp))
    helper.print_V(V_dp, 3, (4,4), helper.SeperatorLines.middle, "DP 算法计算的 V")
    helper.print_V(V, 3, (4,4), helper.SeperatorLines.middle, "MC 算法计算的 V")
    print("Q 误差=",helper.RMSE(Q, Q_dp))
    helper.print_Q(Q_dp, 3, (4,4), helper.SeperatorLines.middle, "DP 算法计算的 Q")
    helper.print_Q(Q, 3, (4,4), helper.SeperatorLines.middle, "MC 算法计算的 Q")

    env.close()
