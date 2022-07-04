
import numpy as np
import MC_102_SafetyDrive_DataModel as env
import time
import Algo_MonteCarlo as algo
import common_helper as helper


if __name__=="__main__":
    np.random.seed(15)
    
    print("---------- First Visit ----------")
    dataModel = env.DataModel()
    start = time.time()
    episodes = 20000        # 计算 50000 次的试验的均值作为数学期望值
    gamma = 1.0
    V = algo.MC_FirstVisit(dataModel, dataModel.S.Start, episodes, gamma)
    print("gamma =", gamma)
    helper.print_V(dataModel, V)
    end = time.time()    
    print("耗时 :", end-start)

    V_groundTruth = env.Matrix(dataModel, 1)
    print("误差 =", helper.RMSE(V, V_groundTruth))
