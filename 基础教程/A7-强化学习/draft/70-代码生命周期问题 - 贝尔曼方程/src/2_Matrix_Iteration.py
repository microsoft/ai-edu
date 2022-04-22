import numpy as np
import CodeLifeCycle_DataModel_P as dm

# 矩阵迭代法
def matrix_iteration(dataModel, gamma):
    print("---矩阵迭代法---")
    V_next = np.zeros(dataModel.N)
    count = 0   # 迭代计数器
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1  # 计数器+1
        V = V_next.copy()   # 准备一个备份，用于比较，检查是否收敛
        V_next = dataModel.R + gamma * np.dot(dataModel.P, V)   # 式 3
        if np.allclose(V_next, V):  # 检查收敛性
            break
    print("迭代次数 :", count)
    return V

def print_V(V):
    print(V)
    vv = np.around(V,3)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))

def check_convergence(dataModel):
    print("迭代100次, 检查状态转移矩阵是否趋近于 0: ")
    P_new = dataModel.P.copy()
    for i in range(100):
        P_new = np.dot(dataModel.P, P_new)
    print(np.around(P_new, 3))

if __name__=="__main__":
    dataModel = dm.DataModel()
    gamma = 1

    V = matrix_iteration(dataModel, gamma)
    print_V(V)

    check_convergence(dataModel)
