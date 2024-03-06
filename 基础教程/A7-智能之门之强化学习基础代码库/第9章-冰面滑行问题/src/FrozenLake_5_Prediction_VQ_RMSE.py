import gymnasium as gym
import common.Algo_DP_PolicyEvaluation as algoPI
import common.Algo_MC_OnPolicy_Predict as algo_MC_FV
import common.CommonHelper as helper
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import FrozenLake_2_DP as dp_VQ


mpl.rcParams['font.sans-serif'] = ['SimHei']  
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
    V_dp, Q_dp = algoPI.calculate_VQ_pi(env, policy, gamma)
    V_MC = None
    Q_MC = None
    Errors_V = []
    Errors_Q = []
    for i in tqdm.trange(repeat):
        pred = algo_MC_FV.MC_EveryVisit_Predict_VQ(env, episodes, gamma, policy)
        v_mc, q_mc = pred.run()
        
        if V_MC is None:
            V_MC = v_mc
        else:
            V_MC += v_mc
        if Q_MC is None:
            Q_MC = q_mc
        else:
            Q_MC += q_mc
        average_v = V_MC / (i+1)
        average_q = Q_MC / (i+1)
        err_v = helper.RMSE(average_v, V_dp)
        err_q = helper.RMSE(average_q, Q_dp)
        Errors_V.append(err_v)
        Errors_Q.append(err_q)

    env.close()
    plt.plot(Errors_V, linestyle="-")
    plt.plot(Errors_Q, linestyle=":")
    plt.legend(["V", "Q"])
    plt.xlabel("重复次数")
    plt.ylabel("RMSE误差")
    plt.grid()
    plt.show()

    # plt.plot(Errors_Q)
    # plt.xlabel("重复次数")
    # plt.ylabel("RMSE误差")
    # plt.grid()
    # plt.show()
    #print(np.allclose(V_dp, average, rtol=1e-3))