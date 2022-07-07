import numpy as np
import tqdm
import math

# MC0 - 多次采样并随采样顺序正向计算 G 值，
# 然后获得回报 G 的数学期望，即状态价值函数 v(start_state)
def MC_Sequential_V(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
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
def MC_FirstVisit_V(dataModel, start_state, episodes, gamma):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        TrajectoryState = []        # 一幕内的状态序列
        TrajectoryReward = []       # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):    # 幕内循环
            next_s, r, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            TrajectoryState.append(s.value)
            TrajectoryReward.append(r)
            s = next_s
        assert(len(TrajectoryState) == len(TrajectoryReward))
        num_step = len(TrajectoryState)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s = TrajectoryState[t]
            r = TrajectoryReward[t]
            G = gamma * G + r
            if not (s in TrajectoryState[0:t]):# 首次访问型
                Value[s] += G     # 值累加
                Count[s] += 1     # 数量加 1
    
    Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    return Value / Count    # 求均值


# MC2-EveryVisit - 每次访问法
def MC_EveryVisit_V(dataModel, start_state, episodes, gamma):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Trajectory = []     # 一幕内的(状态,奖励)序列
        s = start_state
        is_end = False
        while (is_end is False):            # 幕内循环
            next_s, reward, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Trajectory.append((s.value, reward))
            s = next_s

        num_step = len(Trajectory)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, r = Trajectory[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1
    
    Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    return Value / Count    # 求均值
