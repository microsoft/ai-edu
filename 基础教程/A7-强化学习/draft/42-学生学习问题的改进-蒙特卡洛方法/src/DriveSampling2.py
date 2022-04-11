
import tqdm
import multiprocessing as mp
import math
import numpy as np
import DriveDataModel as data
import time


# 多次采样获得回报 G 的数学期望，即状态价值函数 V
def MC_Sampling(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
        # 由于使用了注重结果奖励方式，所以起始状态也有奖励，做为 G 的初始值
        G = dataModel.get_reward(start_state)   
        curr_s = start_state        # 把给定的起始状态作为当前状态
        t = 1                       # 折扣因子
        is_end = False                # 分幕结束标志
        while (is_end is False):      # 本幕循环
            # 根据当前状态和转移概率获得:下一个状态,奖励,是否到达终止状态
            next_s, r, is_end = dataModel.step(curr_s)   
            G += math.pow(gamma, t) * r
            t += 1
            curr_s = next_s
        # end while
        G_sum += G # 先暂时不计算平均值，而是简单地累加
    # end for
    V = G_sum / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return V

# 反向计算G值，记录每个状态的G值，每次访问型
def MC_Sampling_Reverse(dataModel, start_state, episodes, gamma):
    V_value_count_pair = np.zeros((dataModel.num_states, 2))  # state[total value, count of g]
    for episode in tqdm.trange(episodes):
        trajectory = []     # 按顺序 t 保留一幕内的采样序列
        trajectory.append((start_state.value, dataModel.get_reward(start_state)))
        curr_s = start_state
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, r, is_end = dataModel.step(curr_s)
            trajectory.append((next_s.value, r))
            curr_s = next_s
        #endwhile
        G = 0
        # 从后向前遍历
        for t in range(len(trajectory)-1, -1, -1):
            s, r = trajectory[t]
            G = gamma * G + r
            V_value_count_pair[s, 0] += G     # 累积总和
            V_value_count_pair[s, 1] += 1     # 累计次数
        #endfor
    #endfor
    V = V_value_count_pair[:,0] / V_value_count_pair[:,1]   # 计算平均值
    return V

# MC1的改进，反向计算G值，记录每个状态的G值，每次访问型
def MC_Sampling_Reverse_FirstVisit(dataModel, start_state, episodes, gamma):
    V_value_count_pair = np.zeros((dataModel.num_states, 2))  # state[total value, count of g]
    V_value_count_pair[:,1] = 1 # 避免被除数为0
    for episode in tqdm.trange(episodes):
        trajectory = []     # 一幕内的采样序列
        curr_s = start_state
        trajectory.append((curr_s.value, dataModel.get_reward(start_state)))
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, r, is_end = dataModel.step(curr_s)
            #endif
            trajectory.append((next_s.value, r))
            curr_s = next_s
        #endwhile
        num_step = len(trajectory)
        G = 0
        first_visit = set()
        # 从后向前遍历
        for t in range(num_step-1, -1, -1):
            s, r = trajectory[t]
            G = gamma * G + r
            if (s in first_visit):
                continue
            V_value_count_pair[s, 0] += G     # total value
            V_value_count_pair[s, 1] += 1     # count
            first_visit.add(s)
        #endfor
    #endfor
    V = V_value_count_pair[:,0] / V_value_count_pair[:,1]
    return V

def RMSE(a,b):
    err = np.sqrt(np.sum(np.square(a - b))/a.shape[0])
    print("RMSE=",err)

if __name__=="__main__":
    start = time.time()
    episodes = 50000        # 计算 10000 次的试验的均值作为数学期望值
    gammas = [1]    # 指定多个折扣因子做试验
    dataModel = data.DataModel()
    for gamma in gammas:
        V = MC_Sampling_Reverse(dataModel, dataModel.S.Start, episodes, gamma)
        print("gamma =", gamma)
        for s in dataModel.S:
            print(str.format("{0}:\t{1}", s.name, V[s.value]))
    end = time.time()
    print(end-start)
    RMSE(V, dataModel.V_ground_truth)