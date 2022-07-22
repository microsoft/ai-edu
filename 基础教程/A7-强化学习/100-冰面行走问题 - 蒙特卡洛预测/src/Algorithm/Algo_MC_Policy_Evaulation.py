import numpy as np
import tqdm
import math

# MC 策略评估（预测）：每次访问法估算 V_pi
def MC_EveryVisit_V_Policy(env, episodes, gamma, policy, checkpoint=1000, delta=1e-3):
    nS = env.observation_space.n
    nA = env.action_space.n
    Value = np.zeros(nS)  # G 的总和
    Count = np.zeros(nS)  # G 的数量
    V_old = np.zeros(nS)
    for episode in tqdm.trange(episodes):   # 多幕循环
        Episode = []        # 一幕内的(状态,奖励)序列
        s = env.reset()     # 重置环境，开始新的一幕采样
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, _ = env.step(action)
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
            print(np.reshape(np.round(V,3),(4,4)))
            if abs(V-V_old).max() < delta:
                break
            V_old = V.copy()
    print("循环幕数 =",episode+1)
    V = Value / Count    # 求均值
    return V


# MC 策略评估（预测）：每次访问法估算 Q_pi
def MC_EveryVisit_Q_Policy(env, episodes, gamma, policy):
    nA = env.action_space.n                 # 动作空间
    nS = env.observation_space.n            # 状态空间
    Value = np.zeros((nS, nA))              # G 的总和
    Count = np.zeros((nS, nA))              # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s, _ = env.reset(return_info=True)
        Episode = []     # 一幕内的(状态,动作,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, _ = env.step(action)
            Episode.append((s, action, reward))
            s = next_s  # 迭代

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s,a] += G     # 值累加
            Count[s,a] += 1     # 数量加 1

    Count[Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
    Q = Value / Count   # 求均值
    return Q   
