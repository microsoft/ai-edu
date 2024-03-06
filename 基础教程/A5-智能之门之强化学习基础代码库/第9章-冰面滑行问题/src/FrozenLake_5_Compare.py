import gymnasium as gym
import numpy as np
import common.CommonHelper as helper
import common.Algo_DP_PolicyEvaluation as algoPI
import matplotlib as mpl
import FrozenLake_2_DP as dp_VQ
import matplotlib.pyplot as plt
import tqdm


mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False

# 用同一采样过程比较 FirstVisit 和 EveryVisit 的结果
def MC_FirstVist_vs_EveryVisit_VQ(
    env: gym.Env, 
    episodes: int, 
    gamma: float, 
    policy,
) -> np.ndarray:
    nS = env.observation_space.n
    nA = env.action_space.n
    V_first_cum = np.zeros(nS)
    Q_first_cum = np.zeros((nS, nA))
    V_first_cnt = np.zeros(nS)  # G 的数量
    Q_first_cnt = np.zeros((nS, nA)) # G 的数量
    V_every_cum = np.zeros(nS)
    Q_every_cum = np.zeros((nS, nA))
    V_every_cnt = np.zeros(nS)  # G 的数量
    Q_every_cnt = np.zeros((nS, nA)) # G 的数量
    for episode in range(episodes):   # 多幕循环
        Trajactory_state = []        # 一幕内的(状态,奖励)序列
        Trajactory_action = []        # 一幕内的(状态,奖励)序列
        Trajactory_reward = []        # 一幕内的(状态,奖励)序列
        Trajactory_sa = []        
        s, _ = env.reset()     # 重置环境，开始新的一幕采样
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, truncated, _ = env.step(action)
            Trajactory_state.append(s)
            Trajactory_action.append(action)
            Trajactory_reward.append(reward)
            Trajactory_sa.append((s, action))
            s = next_s
        assert(len(Trajactory_state) == len(Trajactory_reward) == len(Trajactory_action) == len(Trajactory_sa))
        # 从后向前遍历计算 G 值
        num_step = len(Trajactory_state)
        G = 0
        for t in range(num_step-1, -1, -1):
            s = Trajactory_state[t]
            a = Trajactory_action[t]
            r = Trajactory_reward[t]
            G = gamma * G + r
            # First Visit
            if not (s in Trajactory_state[0:t]):  # 首次访问型
                V_first_cum[s] += G           # 值累加
                V_first_cnt[s] += 1     # 数量加 1
            if not ((s,a) in Trajactory_sa[0:t]):
                Q_first_cum[s, a] += G        # 值累加
                Q_first_cnt[s, a] += 1     # 数量加 1
            # Every Visit
            V_every_cum[s] += G           # 值累加
            V_every_cnt[s] += 1     # 数量加 1
            Q_every_cum[s, a] += G        # 值累加
            Q_every_cnt[s, a] += 1     # 数量加 1

    V_first_cnt[V_first_cnt==0] = 1 # 把分母为0的填成1，主要是终止状态
    Q_first_cnt[Q_first_cnt==0] = 1 # 把分母为0的填成1，主要是终止状态
    V_every_cnt[V_every_cnt==0] = 1 # 把分母为0的填成1，主要是终止状态
    Q_every_cnt[Q_every_cnt==0] = 1 # 把分母为0的填成1，主要是终止状态
    V_first = V_first_cum / V_first_cnt    # 求均值
    Q_first = Q_first_cum / Q_first_cnt    # 求均值
    V_every = V_every_cum / V_every_cnt    # 求均值
    Q_every = Q_every_cum / Q_every_cnt    # 求均值
    return V_first, Q_first, V_every, Q_every


if __name__=="__main__":
    gamma = 1.0
    episodes = 1000
    repeat = 50
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    env.reset(seed=5)
    # 随机策略
    policy = helper.create_policy(
        env.observation_space.n, env.action_space.n, (0.25, 0.25, 0.25, 0.25))
    # env.reset(seed=5)  # 在下面的函数内部有重置环境的代码
    V_dp, Q_dp = algoPI.calculate_VQ_pi(env, policy, gamma)

    V_FIRST = None
    V_EVERY = None
    V_Errors_FIRST = []
    V_Errors_EVERY = []
    Q_FIRST = None
    Q_EVERY = None
    Q_Errors_FIRST = []
    Q_Errors_EVERY = []

    for i in tqdm.trange(repeat):
        v_first, q_first, v_every, q_every = MC_FirstVist_vs_EveryVisit_VQ(
            env, episodes, gamma, policy
        )

        V_FIRST = v_first if V_FIRST is None else V_FIRST + v_first
        V_EVERY = v_every if V_EVERY is None else V_EVERY + v_every
        Q_FIRST = q_first if Q_FIRST is None else Q_FIRST + q_first
        Q_EVERY = q_every if Q_EVERY is None else Q_EVERY + q_every

        V_Errors_FIRST.append(helper.RMSE(V_FIRST / (i+1), V_dp))
        V_Errors_EVERY.append(helper.RMSE(V_EVERY / (i+1), V_dp))

        Q_Errors_FIRST.append(helper.RMSE(Q_FIRST / (i+1), Q_dp))
        Q_Errors_EVERY.append(helper.RMSE(Q_EVERY / (i+1), Q_dp))

    env.close()

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121)
    ax.plot(V_Errors_FIRST, linestyle="-")
    ax.plot(V_Errors_EVERY, linestyle=":")
    ax.legend(["First Visit", "Every Visit"])
    ax.set_xlabel("幕数")
    ax.set_ylabel("RMSE误差")
    ax.set_title("V 函数")
    ax.grid()
    
    ax = fig.add_subplot(122)
    ax.plot(Q_Errors_FIRST, linestyle="-")
    ax.plot(Q_Errors_EVERY, linestyle=":")
    ax.legend(["First Visit", "Every Visit"])
    ax.set_xlabel("幕数")
    ax.set_ylabel("RMSE误差")
    ax.set_title("Q 函数")
    ax.grid()

    plt.show()
