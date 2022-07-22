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
def MC_EveryVisit_V_Policy_test(env, episodes, gamma, policy, checkpoint=1000, delta=1e-3):
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
            #print(np.reshape(np.round(V,3),(4,4)))
            #if abs(V-V_old).max() < delta:
            #    break
            #V_old = V.copy()
    #print("循环幕数 =",episode+1)
    return V_history    # 返回历史数据用于评测


if __name__=="__main__":
    gamma = 1
    episodes = 10000
    env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
    # 随机策略
    nA = env.action_space.n
    nS = env.observation_space.n
    policy = np.ones(shape=(nS, nA)) / nA   # 随机策略，每个状态上的每个动作都有0.25的备选概率
    # DP
    V_real = get_groud_truth(env, policy, gamma)
    # MC
    start_state, info = env.reset(seed=5, return_info=True)
    # V = algoMC.MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy)
    V_sum = np.zeros(nS)
    V_count = 0
    Errors = []
    for i in range(10):
        Errors.append([])
        V_history = MC_EveryVisit_V_Policy_test(env, episodes, gamma, policy)
        for V in V_history:
            error = helper.RMSE(V, V_real)
            Errors[i].append(error)
            V_sum += V
            V_count += 1
    env.close()
    EArray = np.array(Errors)  
    Errors = np.mean(EArray, axis=0)    

    print("------ 状态价值函数 -----")
    print(np.reshape(np.round(V,3),(4,4)))
    print("误差 =", error)
    plt.plot(Errors)
    plt.title(u'策略评估 $V_\pi$ 的误差与循环次数的关系')
    plt.xlabel(u'循环次数(x1000)')
    plt.ylabel(u'误差 RMSE')
    plt.grid()
    plt.show()

    print("------ 平均值 状态价值函数 -----")
    V_average = V_sum/V_count
    print(V_average)
    print(helper.RMSE(V_average, V_real))
