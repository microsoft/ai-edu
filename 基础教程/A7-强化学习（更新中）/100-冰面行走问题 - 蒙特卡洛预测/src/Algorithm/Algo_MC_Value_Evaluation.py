import numpy as np
import tqdm
import math

# 来自于第六章
# MC0 - 多次采样并随采样顺序正向计算 G 值，
# 然后获得回报 G 的数学期望，即状态价值函数 v(start_state)
def MC_Sequential_V(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for _ in tqdm.trange(episodes):
        s = start_state # 把给定的起始状态作为当前状态
        G = 0           # 设置本幕的初始值 G=0
        t = 0           # 步数计数器
        is_end = False
        while not is_end:
            s_next, reward, is_end = dataModel.step(s)
            G += math.pow(gamma, t) * reward
            t += 1
            s = s_next
        # end while
        G_sum += G # 先暂时不计算平均值，而是简单地累加
    # end for
    v = G_sum / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return v


# MC1-FirstVisit 首次访问法
def MC_FirstVisit_V(dataModel, start_state, episodes, gamma, checkpoint=100, delta=1e-3):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Episode_State = []        # 一幕内的状态序列
        Episode_Reward = []       # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):    # 幕内循环
            next_s, r, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Episode_State.append(s.value)
            Episode_Reward.append(r)
            s = next_s
        assert(len(Episode_State) == len(Episode_Reward))
        num_step = len(Episode_State)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s = Episode_State[t]
            r = Episode_Reward[t]
            G = gamma * G + r
            if not (s in Episode_State[0:t]):# 首次访问型
                Value[s] += G     # 值累加
                Count[s] += 1     # 数量加 1

    Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    return Value / Count    # 求均值


# MC - FirstVisit with EarlyStop 首次访问法,早停
def MC_FirstVisit_EarlyStop(dataModel, start_state, episodes, gamma, checkpoint=100, delta=1e-3):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Episode_State = []        # 一幕内的状态序列
        Episode_Reward = []       # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):    # 幕内循环
            next_s, r, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Episode_State.append(s.value)
            Episode_Reward.append(r)
            s = next_s
        assert(len(Episode_State) == len(Episode_Reward))
        num_step = len(Episode_State)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s = Episode_State[t]
            r = Episode_Reward[t]
            G = gamma * G + r
            if not (s in Episode_State[0:t]):# 首次访问型
                Value[s] += G     # 值累加
                Count[s] += 1     # 数量加 1

        if (episode+1)%checkpoint == 0:
            Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
            V = Value / Count   # 求均值
            if abs(V-V_old).max() < delta:          # 判断是否收敛
                less_than_delta_count += 1          # 计数器 +1
                if (less_than_delta_count == 3):    # 计数器到位
                    break                           # 停止迭代
            else:                                   # 没有持续收敛
                less_than_delta_count = 0           # 计数器复位
            V_old = V.copy()

    return Value / Count    # 求均值


# MC2-EveryVisit - 每次访问法
def MC_EveryVisit_V(dataModel, start_state, episodes, gamma):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for _ in tqdm.trange(episodes):   # 多幕循环
        Episode = []     # 一幕内的(状态,奖励)序列
        s = start_state
        is_end = False
        while (is_end is False):            # 幕内循环
            next_s, reward, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Episode.append((s.value, reward))
            s = next_s

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, r = Episode[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1
    
    Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    return Value / Count    # 求均值
