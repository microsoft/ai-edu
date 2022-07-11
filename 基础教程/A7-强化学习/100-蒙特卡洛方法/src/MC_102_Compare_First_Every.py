
import numpy as np
import MC_102_SafetyDrive_DataModel as env
import common.CommonHelper as helper
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False


# MC2 - 反向计算G值，记录每个状态的G值，首次访问型
def MC_FirstVisit_test(dataModel, start_state, episodes, gamma, checkpoint):
    Vs = []
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Ts = []     # 一幕内的状态序列
        Tr = []     # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):            # 幕内循环
            next_s, r, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Ts.append(s.value)
            Tr.append(r)
            s = next_s
        assert(len(Ts) == len(Tr))
        num_step = len(Ts)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s = Ts[t]
            r = Tr[t]
            G = gamma * G + r
            if not (s in Ts[0:t]):# 首次访问型
                Value[s] += G     # 值累加
                Count[s] += 1     # 数量加 1
    
        if (episode+1)%checkpoint == 0:
            Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
            Vs.append(Value / Count)    # 求均值

    #Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    #Vs.append(Value / Count)    # 求均值
    return Vs


# MC2 - 反向计算G值，记录每个状态的G值，每次访问型
def MC_EveryVisit_test(dataModel, start_state, episodes, gamma, checkpoint):
    Vs = []
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Ts = []     # 一幕内的状态序列
        Tr = []     # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):            # 幕内循环
            next_s, r, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Ts.append(s.value)
            Tr.append(r)
            s = next_s

        num_step = len(Ts)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s = Ts[t]
            r = Tr[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1

        if (episode+1)%checkpoint == 0:
            Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
            Vs.append(Value / Count)    # 求均值

    #Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    #Vs.append(Value / Count)    # 求均值
    return Vs


if __name__=="__main__":
    np.random.seed(5)
    
    dataModel = env.DataModel()
    episodes = 20000        # 计算 20000 次的试验的均值作为数学期望值
    gamma = 1.0
    checkpoint = 100
    Vs_first = MC_FirstVisit_test(dataModel, dataModel.S.Start, episodes, gamma, checkpoint)
    Vs_every = MC_EveryVisit_test(dataModel, dataModel.S.Start, episodes, gamma, checkpoint)
    V_groundTruth = env.Matrix(dataModel, 1)

    errors_first = []
    for v in Vs_first:
        error = helper.RMSE(v, V_groundTruth)
        errors_first.append(error)

    errors_every = []
    for v in Vs_every:
        error = helper.RMSE(v, V_groundTruth)
        errors_every.append(error)

    plt.plot(errors_first[10:], label="首次访问法")
    plt.plot(errors_every[10:], label="每次访问法")
    plt.legend()
    plt.grid()
    plt.title("首次访问法 vs. 每次访问法")
    plt.show()
    