import gymnasium as gym
import numpy as np
import common.Algo_MC_OnPolicy_Predict as algo_MC_FV
import common.CommonHelper as helper
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False


if __name__=="__main__":
    gamma = 1.0
    episodes = 50000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    env.reset(seed=5)
    # 随机策略
    policy = helper.create_policy(
        env.observation_space.n, 
        env.action_space.n, (0.25, 0.25, 0.25, 0.25)
    )
    pred = algo_MC_FV.MC_FirstVisit_Predict_V(env, episodes, gamma, policy)
    V_pi = pred.run() 
    helper.print_V(V_pi, 3, (4,4), helper.SeperatorLines.middle, "状态价值函数")
    env.close()
