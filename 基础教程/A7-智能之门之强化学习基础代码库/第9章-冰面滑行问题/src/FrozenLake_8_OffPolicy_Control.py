import numpy as np
import gymnasium as gym
import common.DrawQpi as DrawQpi
import common.CommonHelper as helper
import common.Algo_DP_ValueIteration as algo_DP
import common.Algo_MC_OffPolicy_Control as algo_MC_OffPolicy_Control
    

if __name__=="__main__":
    env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=True)
    env.reset(seed=5)
    gamma = 0.9
    episodes = 100000
    # 行动策略初始化为随机策略
    behavior_policy = helper.create_policy(
        env.observation_space.n, env.action_space.n, (0.25, 0.25, 0.25, 0.25))
    V_dp, Q_dp = algo_DP.calculate_VQ_star(env, gamma)
    control = algo_MC_OffPolicy_Control.MC_OffPolicy_Control_Policy(
        env, episodes, gamma, behavior_policy)
    Q, target_policy = control.run()
    helper.print_Q(Q, 3, (4,4), helper.SeperatorLines.short, "离策略 MC 控制")
    helper.print_Policy(target_policy, 2, (4,4), helper.SeperatorLines.short, "离策略 MC 控制")
    env.close()
    DrawQpi.drawQ(Q, (4,4), 3, 15, [5,7,11,12])
