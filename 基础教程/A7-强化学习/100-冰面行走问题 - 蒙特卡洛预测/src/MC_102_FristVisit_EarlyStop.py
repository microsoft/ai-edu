
import numpy as np
import MC_102_SafetyDrive_DataModel as env
import common.CommonHelper as helper
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False

# MC2 - 反向计算G值，记录每个状态的G值，首次访问型
def MC_FirstVisit_test(dataModel, start_state, episodes, gamma, checkpoint, delta):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    V_old = np.zeros(dataModel.nS)
    less_than_delta_count = 0
    V_history = []
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
            V = Value / Count   # 求均值
            V_history.append(V)
            #print(np.round(V,3))
            if abs(V-V_old).max() < delta:          # 判定收敛
                less_than_delta_count += 1          # 计数器 +1
                if (less_than_delta_count == 3):    # 计数器到位
                    break                           # 停止迭代
            else:                                   # 没有持续收敛
                less_than_delta_count = 0           # 计数器复位
            V_old = V.copy()
    print("循环幕数 =",episode+1)
    return V_history


if __name__=="__main__":
    np.random.seed(5)
    dataModel = env.DataModel()
    episodes = 100000        # 计算 20000 次的试验的均值作为数学期望值
    gamma = 1.0
    checkpoint = 200
    delta = 1e-2
    V_history = MC_FirstVisit_test(dataModel, dataModel.S.Start, episodes, gamma, checkpoint, delta)
    V_truth = env.Matrix(dataModel, gamma)
    Errors = []
    for V in V_history:
        error = helper.RMSE(V, V_truth)
        Errors.append(error)
    plt.plot(Errors)
    plt.grid()
    plt.xlabel("循环次数(x200)")
    plt.ylabel("RMSE 误差")
    plt.show()

    print("状态值 =", np.round(V,3))
    print("误差 =", error)

