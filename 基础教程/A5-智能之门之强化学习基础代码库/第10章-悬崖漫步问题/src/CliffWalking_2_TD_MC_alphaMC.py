
import gymnasium as gym
import numpy as np
import common.CommonHelper as helper
import tqdm
import common.Algo_DP_PolicyEvaluation as algoPI    
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False

TD = 0
MC = 1
alpha_MC = 2


def compare_TD_MC_alphaMC(
        env: gym.Env, 
        Episodes: int, 
        alpha:float, 
        gamma: float, 
        behavior_policy,
        ground_truth):
    
    V_TD = np.zeros(env.observation_space.n, dtype=np.float32) # 逐步计算 V
    V_MC = np.zeros(env.observation_space.n, dtype=np.float32) # 逐步计算 V
    V_alpha_MC = np.zeros(env.observation_space.n, dtype=np.float32) # 逐步计算 V
    V_count = np.zeros(env.observation_space.n, dtype=np.float32) # 逐步计算 V
    Errors = np.zeros((3, Episodes))    # 三种算法的分幕误差，记录历史便于绘图

    for episode in tqdm.trange(Episodes): # 多幕循环
        state, _ = env.reset()         # 重置环境，开始新的一幕采样
        done = False                        # 一幕结束标志
        Trajectory_reward = []         # 清空 r 轨迹信息
        Trajectory_sa = []             # 清空 s,a 轨迹信息
        Trajectory_state = []          # 清空 s 轨迹信息
        while (done is False):              # 幕内循环
            # 在当前状态 s 下根据策略 policy 选择一个动作
            action = np.random.choice(env.action_space.n, p=behavior_policy[state])
            # 得到下一步的状态、奖励、结束标志、是否超限终止等
            next_state, reward, done, truncated, _ = env.step(action)
            # 记录轨迹信息
            Trajectory_reward.append(reward)
            Trajectory_sa.append((state, action))
            Trajectory_state.append(state)
            # 式（10.2.6）
            V_TD[state] += alpha * (reward + gamma * V_TD[next_state] - V_TD[state])
            state = next_state              # goto next state
        # MC
        num_step = len(Trajectory_reward)
        G = 0
        for t in range(num_step-1, -1, -1):
            s, a = Trajectory_sa[t]
            r = Trajectory_reward[t]
            G = gamma * G + r
            if not (s in Trajectory_state[0:t]):  # 首次访问型
                V_count[s] += 1     # 数量加 1            
                # 式（10.2.2)
                V_MC[s] += (G - V_MC[s]) / V_count[s]     # 值累加
                V_alpha_MC[s] += alpha * (G - V_alpha_MC[s])
        
        Errors[TD, episode] = helper.Norm2Err(V_TD, ground_truth)
        Errors[MC, episode] = helper.Norm2Err(V_MC, ground_truth)
        Errors[alpha_MC, episode] = helper.Norm2Err(V_alpha_MC, ground_truth)
    return Errors, V_TD, V_MC, V_alpha_MC

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

    repeat = 10  # 重复10次
    pool = mp.Pool(processes=4)
    results = []
    for i in range(repeat):
        results.append(pool.apply_async(compare_TD_MC_alphaMC, args=(env, Episodes, alpha, gamma, behavior_policy, V_dp)))
    pool.close()
    pool.join()

    All_Errors = np.zeros((10, 3, Episodes))
    All_Values = np.zeros((10, 3, env.observation_space.n))
    for i in range(len(results)):
        Errors, V_TD, V_MC, V_alpha_MC = results[i].get()
        All_Errors[i] = Errors
        All_Values[i, TD] = V_TD
        All_Values[i, MC] = V_MC
        All_Values[i, alpha_MC] = V_alpha_MC
    
    helper.print_V(np.mean(All_Values[:,TD,:], axis=0), 0, (4,12), helper.SeperatorLines.middle, "TD")
    helper.print_V(np.mean(All_Values[:,MC,:], axis=0), 0, (4,12), helper.SeperatorLines.middle, "MC")
    helper.print_V(np.mean(All_Values[:,alpha_MC,:], axis=0), 0, (4,12), helper.SeperatorLines.middle, "alpha-MC")

    plt.plot(np.mean(All_Errors[:,TD,:], axis=0), label="TD", linestyle="dashed")
    plt.plot(np.mean(All_Errors[:,MC,:], axis=0), label="MC", linestyle="solid")
    plt.plot(np.mean(All_Errors[:,alpha_MC,:], axis=0), label=r'$\alpha$-MC', linestyle="dashdot")
    plt.legend()
    plt.grid()
    plt.xlabel("幕")
    plt.ylabel("误差")
    plt.show()
