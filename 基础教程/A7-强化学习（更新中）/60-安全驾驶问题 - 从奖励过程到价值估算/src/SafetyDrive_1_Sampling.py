
import SafetyDrive_0_DataModel as data
import Algo_Sampling as algo
import time


if __name__=="__main__":
    start = time.time()
    episodes = 10000        # 计算 10000 次的试验的均值作为数学期望值
    gammas = [0, 0.9, 1]    # 折扣因子
    dataModel = data.DataModel()    
    for gamma in gammas:
        V = {}
        for s in dataModel.S:   # 遍历每个状态
            v = algo.Sampling(dataModel, s, episodes, gamma) # 采样计算价值函数
            V[s] = v            # 保存到字典中
        # 打印输出
        print("gamma =", gamma)
        for key, value in V.items():
            print(str.format("{0}:\t{1}", key.name, value))
    end = time.time()
    print("耗时 :", end-start)
