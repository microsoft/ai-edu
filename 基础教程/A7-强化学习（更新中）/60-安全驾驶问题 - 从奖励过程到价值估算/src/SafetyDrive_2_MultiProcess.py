
import multiprocessing as mp
import numpy as np
import SafetyDrive_0_DataModel as data
import Algo_Sampling as algo
import time

def Sampling_MultiProcess(dataModel, episodes, gamma):
    pool = mp.Pool(processes=4) # 指定合适的进程数量
    V = np.zeros((dataModel.nS))
    results = []
    for start_state in dataModel.S:    # 遍历状态集中的每个状态作为起始状态
        results.append(pool.apply_async(algo.Sampling, 
                args=(dataModel, start_state, episodes, gamma,)
            )
        )
    pool.close()
    pool.join()
    for s in range(dataModel.nS):
        v = results[s].get()
        V[s] = v

    return V


if __name__=="__main__":
    start = time.time()
    episodes = 50000        # 计算 10000 次的试验的均值作为数学期望值
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
    print("耗时 :", end-start)
