import numpy as np
import CodeFlow_5_DataModel as data
import common_helper as helper

# 线性方程组双数组迭代法
def double_array_iteration(dataModel, gamma):
    helper.print_seperator_line(helper.SeperatorLines.long, info="线性方程组双数组迭代法")
    V_new = np.zeros(dataModel.N)   # 初始化为全 0
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        V_old = V_new.copy()   # 准备一个备份，用于比较，检查是否收敛
        count += 1  # 计数器+1
        # 列方程组, 更新 V_next 的值
        V_new[0] = dataModel.R[0] + gamma * (0.2 * V_old[0] + 0.8 * V_old[1])
        V_new[1] = dataModel.R[1] + gamma * (0.6 * V_old[0] + 0.4 * V_old[2])
        V_new[2] = dataModel.R[2] + gamma * (0.1 * V_old[1] + 0.9 * V_old[3])
        V_new[3] = dataModel.R[3] + gamma * (0.1 * V_old[1] + 0.2 * V_old[4] + 0.7 * V_old[5])
        V_new[4] = dataModel.R[4] + gamma * (0.2 * V_old[1] + 0.5 * V_old[2] + 0.3 * V_old[3])
        V_new[5] = dataModel.R[5] + gamma * V_old[6]
        V_new[6] = dataModel.R[6]
        if np.allclose(V_new, V_old):  # 检查是否收敛
            break
    print("迭代次数 :", count)
    return V_new

if __name__=="__main__":
    dataModel = data.DataModel()
    gamma = 1
    V = double_array_iteration(dataModel, gamma)
    helper.print_V(dataModel, V)

