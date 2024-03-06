import numpy as np
import gymnasium as gym
import common.Algo_MC_OnPolicy_Predict as algo_MC_FV
import common.CommonHelper as helper
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import common.Algo_DP_PolicyEvaluation as algoPI


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False


if __name__=="__main__":

    gamma = 1.0
    episodes = 1000
    repeat = 50  # 重复次数
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    env.reset(seed=5)    
    # 随机策略
    policy = helper.create_policy(
        env.observation_space.n, 
        env.action_space.n, 
        (0.25, 0.25, 0.25, 0.25)
    )
    V_dp, _ = algoPI.calculate_VQ_pi(env, policy, gamma)
    V_MC = None
    Errors = []
    for i in tqdm.trange(repeat):
        pred = algo_MC_FV.MC_FirstVisit_Predict_V(env, episodes, gamma, policy)
        v_mc = pred.run()
        if V_MC is None:
            V_MC = v_mc  # 首次赋值
        else:
            V_MC += v_mc  # 累加
        v_average = V_MC / (i+1)  # 求平均
        err = helper.RMSE(v_average, V_dp)
        Errors.append(err)

    env.close()
    # plt.semilogy(Errors)
    plt.plot(Errors)
    plt.xlabel("幕数")
    plt.ylabel("RMSE误差")
    plt.title("首次访问法预测 $V$")
    plt.grid()
    plt.show()

    #print(np.allclose(V_dp, average, rtol=1e-3))