import numpy as np
import CodeFlow_5_DataModel as data
import common_helper as helper

# 双数组矩阵迭代法
def matrix_iteration(dataModel, gamma, max_iteration):
    helper.print_seperator_line(helper.SeperatorLines.long, info="双数组矩阵迭代法")
    V_new = np.zeros(dataModel.N)
    count = 0   # 迭代计数器
    while (count < max_iteration):      # 避免不收敛而导致while无限
        count += 1                      # 计数器+1
        V_old = V_new.copy()            # 保存旧值
        V_new = dataModel.R + gamma * np.dot(dataModel.P, V_old)
        if np.allclose(V_new, V_old):   # 检查收敛性
            break
    print("迭代次数 :", count)
    return V_new

def check_convergence(dataModel):
    helper.print_seperator_line(helper.SeperatorLines.long)
    print("转移矩阵自迭代100次, 检查状态转移矩阵是否趋近于 0: ")
    P_new = dataModel.P.copy()
    for i in range(100):
        P_new = np.dot(dataModel.P, P_new)
    print(np.around(P_new, 3))

if __name__=="__main__":
    dataModel = data.DataModel()
    gamma = 1
    max_iteration = 1000
    V = matrix_iteration(dataModel, gamma, max_iteration)
    helper.print_V(dataModel, V)

    check_convergence(dataModel)
