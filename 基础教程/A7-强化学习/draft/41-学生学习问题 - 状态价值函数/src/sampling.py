
import tqdm
import multiprocessing as mp
import math
import numpy as np
import StudentDataModel as data
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


def Sampling(dataModel, start_state, episodes, gamma):
    G_mean = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
        curr_s = start_state        # 把给定的起始状态作为当前状态
        G = dataModel.get_reward(curr_s)   # 由于使用了注重结果奖励方式，所以起始状态也有奖励
        t = 1                   # 折扣因子
        done = False                # 分幕结束标志
        while (done is False):      # 本幕循环
            next_s, r, done = dataModel.step(curr_s)   # 根据当前状态和转移概率获得下一个状态及奖励
            G += math.pow(gamma, t) * r
            t += 1
            curr_s = next_s
        # end while
        G_mean += G # 先暂时不计算平均值，而是简单地累加
    # end for
    v = G_mean / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return v


if __name__=="__main__":
    start = time.time()
    episodes = 10000        # 计算 10000 次的试验的均值作为数学期望值
    gammas = [0, 0.9, 1]    # 指定多个折扣因子做试验
    dataModel = data.DataModel()
    for gamma in gammas:
        V = Sampling_MultiProcess(dataModel, episodes, gamma)
        print("gamma =", gamma)
        for s in dataModel.S:
            print(str.format("{0}:\t{1}", s.name, V[s.value]))
    end = time.time()
    print(end-start)
