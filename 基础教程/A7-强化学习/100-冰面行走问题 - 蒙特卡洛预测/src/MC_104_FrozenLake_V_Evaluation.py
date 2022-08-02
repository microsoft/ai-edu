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
mpl.rcParams['axes.unicode_minus']=False


def get_groud_truth(env, policy, gamma):
    iteration = 100
    V, _ = algoDP.calculate_VQ_pi(env, policy, gamma, iteration)
    return V

# MC 策略评估（预测）：每次访问法估算 V_pi
def MC_EveryVisit_V_Policy_test(env, episodes, gamma, policy, checkpoint=1000):
    nS = env.observation_space.n
    nA = env.action_space.n
    Value = np.zeros(nS) # G 的总和
    Count = np.zeros(nS) # G 的数量
    V_old = np.zeros(nS)
    V_history = []       # 测试用
    for episode in tqdm.trange(episodes):   # 多幕循环
        Episode = []     # 保存一幕内的(状态,奖励)序列
        s, _ = env.reset(return_info=True)# 重置环境，开始新的一幕采样
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, info = env.step(action)
            Episode.append((s, reward))
            s = next_s
        # 从后向前遍历计算 G 值
        G = 0
        for t in range(len(Episode)-1, -1, -1):
            s, r = Episode[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1
        # 检查是否收敛
        if (episode + 1)%checkpoint == 0: 
            Count[Count==0] = 1 # 把分母为0的填成1，主要是对终止状态
            V = Value / Count
            V_history.append(V)
    return V_history    # 返回历史数据用于评测


if __name__=="__main__":
    gamma = 1
    episodes = 10000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    # 随机策略
    policy = helper.create_policy(
        env.observation_space.n, env.action_space.n, (0.25, 0.25, 0.25, 0.25))
    # DP
    V_truth = get_groud_truth(env, policy, gamma)
    # MC
    start_state, info = env.reset(seed=5, return_info=True)
    # V = algoMC.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    checkpoint = 100
    runs = 10
    count = int(episodes/checkpoint)
    V_sum = []              # 准备数据计算平均值的误差
    Errors_history = []     # 准备数据计算误差的平均值
    for i in range(runs):   # 运行 n 次的平均值
        V_sum.append([])
        Errors_history.append([])   # 增加一维空列表，用于存储历史误差数据
        V_history = MC_EveryVisit_V_Policy_test(env, episodes, gamma, policy, checkpoint=checkpoint)
        for j in range(len(V_history)):
            V = V_history[j]
            error = helper.RMSE(V, V_truth)
            Errors_history[i].append(error)
            V_sum[i].append(V)      # 同时计算 V 的平均值
    env.close()

    # n 次 run 的在每个checkpoint上的平均值的误差
    V_average = np.array(V_sum).mean(axis=0)
    rmses = np.zeros(count)
    for i in range(count):
        rmses[i] = helper.RMSE(V_average[i], V_truth)
    # 显示平均值的误差
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)  
    ax1.plot(rmses)
    ax1.set_xlabel(u"循环次数(x1000)")
    ax1.set_ylabel(u"RMSE 误差")
    ax1.set_title(u'平均值的误差')
    ax1.grid()
    
    helper.print_seperator_line(helper.SeperatorLines.long, "平均值的误差")
    helper.print_seperator_line(helper.SeperatorLines.short, "状态价值 V 的平均值")
    print(np.around(V_average[-1], 3).reshape(4,4))
    helper.print_seperator_line(helper.SeperatorLines.short, "状态价值 V 的平均值的误差")
    print(rmses[-1])

    # n 次 run 的在每个checkpoint上的误差的平均值
    Errors = np.array(Errors_history).mean(axis=0)
    helper.print_seperator_line(helper.SeperatorLines.long, "误差的平均值")
    helper.print_seperator_line(helper.SeperatorLines.short, "最后一个状态价值 V")
    print(np.around(V,3).reshape(4,4))
    helper.print_seperator_line(helper.SeperatorLines.short, "最后一个状态价值 V 的误差")
    print(Errors[-1])

    # 显示误差的平均值
    ax2 = fig.add_subplot(122)
    ax2.plot(Errors)
    ax2.set_title(u'误差的平均值')
    ax2.set_xlabel(u'循环次数(x1000)')
    ax2.set_ylabel(u'RMSE 误差')
    ax2.grid()

    plt.show()
