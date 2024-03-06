import numpy as np
import tqdm
import math

# 多次采样获得回报 G 的数学期望，即状态价值函数 v(start_state)
def Sampling(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
        s = start_state # 把给定的起始状态作为当前状态
        G = 0           # 设置本幕的初始值为起始状态的奖励值
        t = 0           # 步数计数器
        is_done = False
        while not is_done:
            s_next, reward, is_done = dataModel.step(s)
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
