import numpy as np
import tqdm
import math


# MC2-EveryVisit - 每次访问法
def MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy):
    nS = env.observation_space.n
    nA = env.action_space.n
    Value = np.zeros(nS)  # G 的总和
    Count = np.zeros(nS)  # G 的数量

    for episode in tqdm.trange(episodes):   # 多幕循环
        Trajectory = []     # 一幕内的(状态,奖励)序列
        s = start_state
        done = False
        while (done is False):            # 幕内循环
            #action = env.action_space.sample()
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, info = env.step(action)
            Trajectory.append((s, reward))
            s = next_s

        #print(Trajectory)
        num_step = len(Trajectory)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, r = Trajectory[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1

        # 重置环境，开始新的一幕采样
        s, info = env.reset(return_info=True)

    Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    V = Value / Count
    return V    # 求均值


# MC2-EveryVisit - 每次访问法
def MC_EveryVisit_Q(env, start_state, episodes, gamma, policy):
    nA = env.action_space.n
    nS = env.observation_space.n
    Value = np.zeros((nS, nA))  # G 的总和
    Count = np.zeros((nS, nA))  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s, info = env.reset(return_info=True)
        Trajectory = []     # 一幕内的(状态,奖励)序列
        s = start_state
        done = False
        while (done is False):            # 幕内循环
            action = env.action_space.sample()  # 随机策略
            next_s, reward, done, info = env.step(action)
            Trajectory.append((s, action, reward))
            s = next_s

        num_step = len(Trajectory)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Trajectory[t]
            G = gamma * G + r
            Value[s,a] += G     # 值累加
            Count[s,a] += 1     # 数量加 1

    Count[Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
    return Value / Count    # 求均值


# MC3 - 增量更新
def MC_Incremental_Update(dataModel, start_state, episodes, alpha, gamma):
    V = np.zeros((dataModel.num_states))
    for episode in tqdm.trange(episodes):
        trajectory = []
        if (start_state is None):
            curr_s = dataModel.random_select_state()
        else:
            curr_s = start_state
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, R, is_end = dataModel.step(curr_s)
            trajectory.append((next_s.value, R))
            curr_s = next_s

        # calculate G_t
        G = 0
        # 从后向前遍历
        for t in range(len(trajectory)-1, -1, -1):
            S, R = trajectory[t]
            G = gamma * G + R
            V[S] = V[S] + alpha * (G - V[S])
        #endfor
    #endfor
    return V