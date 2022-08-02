
import numpy as np
import MC_102_SafetyDrive_DataModel as env
import time
import Algorithm.Algo_MC_Value_Evaluation as algo
import common.CommonHelper as helper


def matrix_method(dataModel, gamma):
    helper.print_seperator_line(helper.SeperatorLines.long, "矩阵法")
    V_truth = env.Matrix(dataModel, gamma)
    print(np.round(V_truth, 2))
    return V_truth


def sequential_method(dataModel, gamma):
    helper.print_seperator_line(helper.SeperatorLines.long, "顺序计算法")
    start = time.time()
    episodes = 5000         # 计算 5000 次循环的均值作为数学期望值
    V = []
    for s in dataModel.S:   # 遍历每个状态
        v = algo.MC_Sequential_V(dataModel, s, episodes, gamma) # 采样计算价值函数
        V.append(v)        # 保存到字典中
    # 打印输出
    helper.print_V(dataModel, np.array(V))
    end = time.time()
    helper.print_seperator_line(helper.SeperatorLines.middle, "耗时")
    print("duration =", end-start)
    return V


def first_visit_method(dataModel, gamma):
    helper.print_seperator_line(helper.SeperatorLines.long, "首次访问法")
    start = time.time()
    episodes = 20000        # 计算 50000 次的试验的均值作为数学期望值
    V = algo.MC_FirstVisit_V(dataModel, dataModel.S.Start, episodes, gamma)
    helper.print_V(dataModel, V)
    end = time.time()    
    helper.print_seperator_line(helper.SeperatorLines.middle, "耗时")
    print("duration =", end-start)
    return V


if __name__=="__main__":
    np.random.seed(15)
    gamma = 1.0
    print("gamma =", gamma)
    dataModel = env.DataModel()

    V_truth = matrix_method(dataModel, gamma)

    V_seq = sequential_method(dataModel, gamma)
    helper.print_seperator_line(helper.SeperatorLines.middle, "顺序计算法与矩阵法之间的误差")
    print("RMSE =", helper.RMSE(V_seq, V_truth))

    V_firstVisit = first_visit_method(dataModel, gamma)
    helper.print_seperator_line(helper.SeperatorLines.middle, "首次访问法与矩阵法之间的误差")
    print("RMSE =", helper.RMSE(V_firstVisit, V_truth))
