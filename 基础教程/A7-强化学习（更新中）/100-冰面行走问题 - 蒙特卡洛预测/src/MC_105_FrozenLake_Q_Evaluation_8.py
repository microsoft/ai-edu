import numpy as np
import gym
import Algorithm.Algo_MC_Policy_Evaulation as algoMC
import Algorithm.Algo_PolicyValueFunction as algoDP
import common.DrawQpi as drawQ
import common.CommonHelper as helper
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False

def test_n_times(env, episodes, gamma, policy, checkpoint=1000, n=10):
    Errors = []
    for i in range(n):
        Q_history, T_len = MC_EveryVisit_Q_Policy_test(env, episodes, gamma, policy, checkpoint=checkpoint)
        errors = []
        for Q in Q_history:
            error = helper.RMSE(Q, Q_real)
            errors.append(error)
        Errors.append(errors)
    E_array = np.array(Errors)
    E_mean = np.mean(E_array, axis=0)   # 求 n 次的误差平均值
    return E_mean, T_len, Q, error


# MC 策略评估（预测）：每次访问法估算 Q_pi
def MC_EveryVisit_Q_Policy_test(env, episodes, gamma, policy, checkpoint=1000):
    nA = env.action_space.n
    nS = env.observation_space.n
    Value = np.zeros((nS, nA))  # G 的总和
    Count = np.zeros((nS, nA))  # G 的数量
    Q_history = []
    T_len = 0
    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s = env.reset()
        Episode = []     # 一幕内的(状态,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, _ = env.step(action)
            Episode.append((s, action, reward))
            s = next_s
        
        T_len += len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(len(Episode)-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s,a] += G     # 值累加
            Count[s,a] += 1     # 数量加 1

        if (episode + 1)%checkpoint == 0: 
            Count[Count==0] = 1 # 把分母为0的填成1，主要是对终止状态
            Q = Value / Count
            Q_history.append(Q)
    return Q_history, T_len/episodes

def get_groud_truth(env, policy, gamma):
    iteration = 100
    _, Q = algoDP.calculate_VQ_pi(env, policy, gamma, iteration)
    return Q


if __name__=="__main__":
    gamma = 0.9
    episodes = 20000
    policy_names = ["正确策略", "随机策略","错误策略"]
    policies = [
        # left, down, right, up
        (0.2,  0.3,  0.3,  0.2),
        (0.25, 0.25, 0.25, 0.25), 
        (0.3,  0.2,  0.2,  0.3)
    ]
    end_states = [19, 29, 35, 41, 42, 46, 49, 52, 54, 59, 63]
    np.set_printoptions(suppress=True)

    for i, policy_data in enumerate(policies):
        helper.print_seperator_line(helper.SeperatorLines.long, policy_names[i])
        env = gym.make("FrozenLake-v1", desc=None, map_name = "8x8", is_slippery=False)
        nA = env.action_space.n
        nS = env.observation_space.n
        policy = helper.create_policy(nS, nA, policy_data)
        helper.print_seperator_line(helper.SeperatorLines.short, "策略")
        print(policy)
        Q_real = get_groud_truth(env, policy, gamma)
        start_state, info = env.reset(seed=5, return_info=True)
        Errors, T_len, Q, error = test_n_times(env, episodes, gamma, policy, checkpoint=1000, n=1)    # n=10
        helper.print_seperator_line(helper.SeperatorLines.short, "动作价值函数")
        print(np.round(Q,3))
        helper.print_seperator_line(helper.SeperatorLines.short, "误差")
        print("RMSE =", error)
        helper.print_seperator_line(helper.SeperatorLines.short, "平均每幕长度")
        print("Len =", T_len)
        plt.plot(Errors, label=policy_names[i])
        env.close()
        policy = helper.extract_policy_from_Q(Q, end_states)
        helper.print_seperator_line(helper.SeperatorLines.short, "抽取出来的策略")
        print(policy)
        drawQ.draw(policy,(8,8))

    plt.title(u'三种策略的误差收敛情况比较')
    plt.xlabel(u'循环次数(x1000)')
    plt.ylabel(u'误差 RMSE')
    plt.legend()
    plt.grid()
    plt.show()
