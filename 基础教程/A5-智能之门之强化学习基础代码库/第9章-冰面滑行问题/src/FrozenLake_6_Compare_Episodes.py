import numpy as np
import gymnasium as gym
import common.CommonHelper as helper
import common.Algo_MC_OnPolicy_Control as algo_MC_Control
import common.Algo_DP_ValueIteration as algo_DP_VI


def test_MC_Control(env, gamma, episodes, epsilon, round=4):
    helper.print_seperator_line(helper.SeperatorLines.long, "epsilon = "+str(epsilon))
    # 初始化为随机策略（可以是别的策略）
    policy = helper.create_policy(
        env.observation_space.n, env.action_space.n, (0.25, 0.25, 0.25, 0.25))

    control = algo_MC_Control.MC_FirstVisit_Control_Greedy(env, episodes, gamma, policy)
    control.set_greedy_fun(algo_MC_Control.MC_Soft_Greedy, epsilon)
    V_star, Q_star, best_policy = control.run()
    
    np.set_printoptions(suppress=True)
    helper.print_Q(best_policy, round, (4,4), helper.SeperatorLines.middle, "最终策略")
    helper.print_V(V_star, round, (4,4), helper.SeperatorLines.middle, "状态价值")
    helper.print_Q(Q_star, round, (4,4), helper.SeperatorLines.middle, "动作价值")

    V_dp, Q_dp = algo_DP_VI.calculate_VQ_star(env, gamma)
    helper.print_seperator_line(helper.SeperatorLines.middle, "V* 的误差")
    err_v = helper.RMSE(V_star, V_dp)
    print(err_v)
    helper.print_seperator_line(helper.SeperatorLines.middle, "Q* 的误差")
    err_q = helper.RMSE(Q_star, Q_dp)
    print(err_q)

    return best_policy, err_v, err_q, V_star, Q_star


if __name__=="__main__":
    env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=True)
    env.reset(seed=5)
    gamma = 0.9
    # 测试不同 episodes 的误差
    test_episodes = [10000,50000,100000]
    epsilon = 0.3
    for episodes in test_episodes:  # 测试不同 episodes 的误差
        test_MC_Control(env, gamma, episodes, epsilon)
    
    env.close()
    
