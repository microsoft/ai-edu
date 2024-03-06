import numpy as np
import gymnasium as gym
import common.DrawQpi as drawQ
import common.CommonHelper as helper
import common.Algo_MC_OnPolicy_Control as algo_MC_Control


if __name__=="__main__":
    env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=True)
    env.reset(seed=5)
    gamma = 0.9
    episodes = 50000

    # 初始化为随机策略（可以是别的软性策略）
    policy = helper.create_policy(
        env.observation_space.n, env.action_space.n, (0.25, 0.25, 0.25, 0.25))

    control = algo_MC_Control.MC_FirstVisit_Control_Greedy(env, episodes, gamma, policy)
    control.set_greedy_fun(algo_MC_Control.MC_Greedy)
    V, Q, final_policy = control.run()
    
    helper.print_Q(final_policy, 3, (4,4), helper.SeperatorLines.middle, "最终策略")
    #drawQ.drawPolicy(best_policy, (4,4), round=4, goal_state=15)    

    helper.print_V(V, 3, (4,4), helper.SeperatorLines.middle, "状态价值")

    np.set_printoptions(suppress=True)
    helper.print_Q(Q, 3, (4,4), helper.SeperatorLines.middle, "动作价值")
    drawQ.drawQ(Q, (4,4), round=3, goal_state=15, end_state=[5,7,11,12])    
