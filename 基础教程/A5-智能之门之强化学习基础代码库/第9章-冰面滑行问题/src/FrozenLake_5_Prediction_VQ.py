import gymnasium as gym
import common.Algo_MC_OnPolicy_Predict as algo_MC_FV
import common.CommonHelper as helper
import matplotlib as mpl
import common.DrawQpi as drawQ

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False


if __name__=="__main__":
    gamma = 1.0
    episodes = 50000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    env.reset(seed=5)
    # 随机策略
    policy = helper.create_policy(
        env.observation_space.n, env.action_space.n, (0.25, 0.25, 0.25, 0.25))
    pred = algo_MC_FV.MC_EveryVisit_Predict_VQ(env, episodes, gamma, policy)
    V_pi, Q_pi = pred.run()
    helper.print_V(V_pi, 3, (4,4), helper.SeperatorLines.middle, "随机状态下的状态价值函数 V")
    helper.print_Q(Q_pi, 3, (4,4), helper.SeperatorLines.middle, "随机状态下的动作价值函数 Q")
    drawQ.drawQ(Q_pi, (4,4), round=3, goal_state=15, end_state=[5,7,11,12])
    
    env.close()
