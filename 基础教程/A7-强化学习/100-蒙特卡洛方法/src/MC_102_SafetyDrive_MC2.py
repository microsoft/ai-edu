
import numpy as np
import MC_102_SafetyDrive_DataModel as env
import time
import Algo_MonteCarlo as algo

def RMSE(a,b):
    err = np.sqrt(np.sum(np.square(a - b))/b.shape[0])
    return err

if __name__=="__main__":
    np.random.seed(15)
    
    print("---------- MC-2 法 First Visit ----------")
    dataModel = env.DataModel()
    start = time.time()
    episodes = 50000        # 计算 10000 次的试验的均值作为数学期望值
    gamma = 1.0
    V2 = algo.MC_FirstVisit(dataModel, dataModel.S.Start, episodes, gamma)
    print("gamma =", gamma)
    for s in dataModel.S:
        print(str.format("{0}:\t{1:.2f}", s.name, V2[s.value]))
    end = time.time()    
    print("耗时 :", end-start)

    print("---------- MC-1 法 Sequential ----------")
    start = time.time()
    episodes = 10000        # 计算 10000 次的试验的均值作为数学期望值
    V1 = {}
    for s in dataModel.S:   # 遍历每个状态
        v = algo.MC_Sequential(dataModel, s, episodes, gamma) # 采样计算价值函数
        V1[s] = v            # 保存到字典中
    # 打印输出
    for key, value in V1.items():
        print(str.format("{0}:\t{1:.2f}", key.name, value))
    end = time.time()
    print("耗时 :", end-start)

    print("---------- 贝尔曼方程法 ----------")
    V_groundTruth = env.Matrix(dataModel, 1)
    print("MC-2 误差 =", RMSE(V2, V_groundTruth))
    print("MC-1 误差 =", RMSE(list(V1.values()), V_groundTruth))
