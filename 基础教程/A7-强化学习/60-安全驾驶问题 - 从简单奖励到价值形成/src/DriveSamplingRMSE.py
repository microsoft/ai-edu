
import tqdm
import multiprocessing as mp
import math
import numpy as np
import DriveDataModel as data
import time
import matplotlib.pyplot as plt

def Sampling_MultiProcess(dataModel, episodes, gamma, checkpoint):
    pool = mp.Pool(processes=4) # 指定合适的进程数量
    V = dict()
    results = []
    for start_state in dataModel.S:    # 遍历状态集中的每个状态作为起始状态
        results.append(pool.apply_async(Sampling_Checkpoint, 
                args=(dataModel, start_state, episodes, gamma, checkpoint,)
            )
        )
    pool.close()
    pool.join()
    for s in range(dataModel.num_states):
        v = results[s].get()
        V[s] = v

    return V

# 多次采样获得回报 G 的数学期望，即状态价值函数 V
def Sampling_Checkpoint(dataModel, start_state, episodes, gamma, checkpoint):
    V = []
    G_sum = 0  # 定义最终的返回值，G 的平均数
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
        if (episode+1)%checkpoint == 0:
            V.append(G_sum / (episode+1))
    # end for
    V.append(G_sum / episodes)
    return V


def RMSE(a,b):
    err = np.sqrt(np.sum(np.square(a - b))/a.shape[0])
    return err

def test_once():
    episodes = 10000        # 计算 10000 次的试验的均值作为数学期望值
    gamma = 1   
    checkpoint = 100
    dataModel = data.DataModel()
    V = Sampling_MultiProcess(dataModel, episodes, gamma, checkpoint)
    num_checkpoint = len(V[0])
    array = np.zeros((num_checkpoint, dataModel.num_states))
    for s_value in range(dataModel.num_states):
        array[:,s_value] = V[s_value]     # V is dictionary
    
    errors = []
    for i in range(num_checkpoint):
        err = RMSE(array[i], dataModel.V_ground_truth)
        errors.append(err)

    return errors

if __name__=="__main__":
    ERRORS = []
    for i in range(10):
        errors = test_once()
        ERRORS.append(errors)

    avg_E = np.mean(ERRORS, axis=0)
    plt.plot(avg_E)    
    plt.grid()
    plt.title(str.format("min RMSE={0}", np.min(avg_E)))
    plt.show()
    