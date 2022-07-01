import numpy as np
import tqdm
import math

# MC1 - 多次采样并随采样顺序正向计算 G 值，
# 然后获得回报 G 的数学期望，即状态价值函数 v(start_state)
def MC_Sequential(dataModel, start_state, episodes, gamma):
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

def print_V(dataModel, V):
    vv = np.around(V,2)
    print("状态价值函数计算结果(数组) :", vv)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))


# MC2 - 反向计算G值，记录每个状态的G值，首次访问型
def MC_FirstVisit(dataModel, start_state, episodes, gamma):
    Value = np.zeros(dataModel.nS)
    Count = np.zeros(dataModel.nS)
    #V_value_count_pair[:,1] = 1 # 避免被除数为0
    for episode in tqdm.trange(episodes):
        Ts = []     # 一幕内的状态序列
        Tr = []     # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, r, is_end = dataModel.step(s)
            Ts.append(s.value)
            Tr.append(r)
            s = next_s
        #endwhile
        num_step = len(Ts)
        G = 0
        # 从后向前遍历
        for t in range(num_step-1, -1, -1):
            s = Ts[t]
            r = Tr[t]
            G = gamma * G + r
            if not (s in Ts[0:t]):
                Value[s] += G     # total value
                Count[s] += 1     # count
        #endfor
    #endfor
    # 把分母为0的填成1
    Count[Count==0] = 1
    return Value / Count
