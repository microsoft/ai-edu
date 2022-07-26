import numpy as np
import tqdm
import math

# MC 策略评估（预测）：每次访问法估算 V_pi
def MC_EveryVisit_V_Policy(env, episodes, gamma, policy):
    nS = env.observation_space.n
    nA = env.action_space.n
    Value = np.zeros(nS)  # G 的总和
    Count = np.zeros(nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Episode = []     # 一幕内的(状态,奖励)序列
        s = env.reset() # 重置环境，开始新的一幕采样
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
    Count[Count==0] = 1 # 把分母为0的填成1，主要是对终止状态
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
        s = env.reset()
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


def get_start_state(env, start_state):
    s = env.reset()
    while s != start_state:     # 以随机选择的 s 为起始状态
        action = np.random.choice(env.action_space.n)
        next_s, _, done, _ = env.step(action)
        s = next_s
        if s == start_state:
            break
        if done is True:
            s = env.reset()
    return s

# MC 控制 探索出发
def MC_ES(env, episodes, gamma, policy):
    nA = env.action_space.n
    nS = env.observation_space.n
    Value = np.zeros((nS, nA))  # G 的总和
    Count = np.zeros((nS, nA))  # G 的数量

    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        start_state = np.random.choice(nS)
        s = get_start_state(env, start_state)
        assert(s == start_state)
        # 找到了指定的 start_state，开始采样
        Trajectory = []     # 一幕内的(状态,奖励)序列
        action = np.random.choice(nA)   # 起始动作也是随机的
        next_s, reward, done, _ = env.step(action)
        Trajectory.append((s, action, reward))
        s = next_s
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])  # 根据改进的策略采样
            next_s, reward, done, _ = env.step(action)
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
            policy[s] = 0
            policy[s,A] = 1

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