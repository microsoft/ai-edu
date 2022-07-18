
import numpy as np
import MC_102_SafetyDrive_DataModel as env
import time
import Algorithm.Algo_MonteCarlo as algo
import common.CommonHelper as helper


if __name__=="__main__":
    np.random.seed(15)
    print("---------- Sequential ----------")   
    dataModel = env.DataModel()
    start = time.time()
    gamma = 1.0
    start = time.time()
    episodes = 5000        # 计算 5000 次的试验的均值作为数学期望值
    V1 = {}
    for s in dataModel.S:   # 遍历每个状态
        v = algo.MC_Sequential(dataModel, s, episodes, gamma) # 采样计算价值函数
        V1[s] = v            # 保存到字典中
    # 打印输出
    for key, value in V1.items():
        print(str.format("{0}:\t{1:.2f}", key.name, value))
    end = time.time()
    print("耗时 :", end-start)

    V_groundTruth = env.Matrix(dataModel, 1)
    print("误差 =", helper.RMSE(list(V1.values()), V_groundTruth))
