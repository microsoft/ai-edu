
import tqdm
import multiprocessing as mp
import math
import numpy as np
import DriveDataModel as data
import time

def Sampling_MultiProcess(dataModel, episodes, gamma):
    pool = mp.Pool(processes=4) # 指定合适的进程数量
    V = np.zeros((dataModel.num_states))
    results = []
    for start_state in dataModel.S:    # 遍历状态集中的每个状态作为起始状态
        results.append(pool.apply_async(Sampling, 
                args=(dataModel, start_state, episodes, gamma,)
            )
        )
    pool.close()
    pool.join()
    for s in range(dataModel.num_states):
        v = results[s].get()
        V[s] = v

    return V

# 多次采样获得回报 G 的数学期望，即状态价值函数 V
def Sampling(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
        s = start_state # 把给定的起始状态作为当前状态
        G = 0           # 设置本幕的初始值 G=0
        t = 0           # 步数计数器
        while True:
            r = dataModel.get_reward(s)
            G += math.pow(gamma, t) * r
            t += 1
            s = dataModel.get_next(s)
            if (s is None):
                break
        # end while
        G_sum += G # 先暂时不计算平均值，而是简单地累加
    # end for
    V = G_sum / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return V

'''
# 多次采样获得回报 G 的数学期望，即状态价值函数 V
def Sampling(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值, G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
        # 由于使用了面向结果奖励方式，所以起始状态也有奖励，做为 G 的初始值
        G = dataModel.get_reward(start_state)   
        curr_s = start_state        # 把给定的起始状态作为当前状态
        t = 1                       # 折扣因子
        done = False                # 分幕结束标志
        while (done is False):      # 本幕循环
            # 根据当前状态和转移概率获得:下一个状态,奖励,是否到达终止状态
            next_s, r, done = dataModel.step(curr_s)   
            G += math.pow(gamma, t) * r
            t += 1
            curr_s = next_s
        # end while
        G_sum += G # 先暂时不计算平均值，而是简单地累加
    # end for
    V = G_sum / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return V
'''
def RMSE(a,b):
    err = np.sqrt(np.sum(np.square(a - b))/a.shape[0])
    return err


if __name__=="__main__":
    start = time.time()
    episodes = 10000        # 计算 10000 次的试验的均值作为数学期望值
    gammas = [0, 0.9, 1]    # 指定多个折扣因子做试验
    Vs = []
    dataModel = data.DataModel()
    for gamma in gammas:
        V = Sampling_MultiProcess(dataModel, episodes, gamma)
        Vs.append(V)
        print("gamma =", gamma)
        for s in dataModel.S:
            print(str.format("{0}:\t{1}", s.name, V[s.value]))
    end = time.time()
    #print(end-start)
    print("RMSE = ", RMSE(Vs[2], dataModel.V_ground_truth))
