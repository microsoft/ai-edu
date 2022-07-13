import numpy as np
import tqdm
import math

# MC 策略评估（预测）：每次访问法估算 V_pi
def MC_EveryVisit_V_Policy(env, start_state, episodes, gamma, policy, checkpoint=1000, delta=1e-3):
    nS = env.observation_space.n
    nA = env.action_space.n
    Value = np.zeros(nS)  # G 的总和
    Count = np.zeros(nS)  # G 的数量
    V_old = np.zeros(nS)
    for episode in tqdm.trange(episodes):   # 多幕循环
        Episode = []     # 一幕内的(状态,奖励)序列
        s = start_state
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
        # 重置环境，开始新的一幕采样
        start_state, _ = env.reset(return_info=True)
        # 检查是否收敛
        if (episode + 1)%checkpoint == 0:
            Count[Count==0] = 1 # 把分母为0的填成1，主要是对终止状态
            V = Value / Count
            print(np.reshape(np.round(V,3),(4,4)))
            if abs(V-V_old).max() < delta:
                break
            V_old = V.copy()
    print("循环幕数 =",episode+1)
    return V    # 求均值


# MC 策略评估（预测）：每次访问法估算 Q_pi
def MC_EveryVisit_Q_Policy(env, start_state, episodes, gamma, policy):
    nA = env.action_space.n
    nS = env.observation_space.n
    Value = np.zeros((nS, nA))  # G 的总和
    Count = np.zeros((nS, nA))  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s, info = env.reset(return_info=True)
        Episode = []     # 一幕内的(状态,奖励)序列
        s = start_state
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, info = env.step(action)
            Episode.append((s, action, reward))
            s = next_s

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

# MC2-EveryVisit - 每次访问法
def MC_EveryVisit_Q_E_Greedy(env, start_state, episodes, gamma, policy, epsilon):
    nA = env.action_space.n
    nS = env.observation_space.n
    Value = np.zeros((nS, nA))  # G 的总和
    Count = np.zeros((nS, nA))  # G 的数量

    other_p = epsilon / nA
    best_p = 1 - epsilon + epsilon/nA

    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s, info = env.reset(return_info=True)
        Trajectory = []     # 一幕内的(状态,奖励)序列
        s = start_state
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
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

        Count[Count==0] = 1
        Q = Value / Count   # 求均值
        
        for s in range(nS):
            max_A = np.max(Q[s])
            argmax_A = np.where(Q[s] == max_A)[0]
            A = np.random.choice(argmax_A)
            policy[s] = other_p
            policy[s,A] = best_p

    Count[Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
    Q = Value / Count   # 求均值
    return Q   



# GLIE - Greedy in the Limit with Infinite Exploration
def MC_EveryVisit_Q_GLIE(env, start_state, episodes, gamma, policy):
    nA = env.action_space.n
    nS = env.observation_space.n
    Q = np.zeros((nS, nA))  # Q 动作价值
    N = np.zeros((nS, nA))  # N 次数

    for k in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s, info = env.reset(return_info=True)
        Trajectory = []     # 一幕内的(状态,奖励)序列
        s = start_state
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, info = env.step(action)
            Trajectory.append((s, action, reward))
            s = next_s

        num_step = len(Trajectory)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Trajectory[t]
            G = gamma * G + r
            N[s,a] += 1     # 数量加 1
            Q[s,a] += (G - Q[s,a])/N[s,a]

        epsilon = 1 / (math.log(k+1)+1)
        other_p = epsilon / nA
        best_p = 1 - epsilon + epsilon/nA


        # 更新策略        
        for s in range(nS):
            max_A = np.max(Q[s])
            argmax_A = np.where(Q[s] == max_A)[0]
            A = np.random.choice(argmax_A)
            policy[s] = other_p
            policy[s,A] = best_p

    return Q

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