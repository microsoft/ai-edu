import numpy as np
import CodeLifeCycle_DataModel_P as dm

# 原始迭代法
def linear_equations_iteration(dataModel, gamma):
    print("---原始迭代法---")
    V_next = np.zeros(dataModel.N)   # 初始化为全 0
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        V = V_next.copy()   # 准备一个备份，用于比较，检查是否收敛
        count += 1  # 计数器+1
        # 列方程组, 更新 V_next 的值
        V_next[0] = dataModel.R[0] + gamma*(0.7 * V[0] + 0.3 * V[1])
        V_next[1] = dataModel.R[1] + gamma*(0.6 * V[0] + 0.4 * V[2])
        V_next[2] = dataModel.R[2] + gamma*(0.9 * V[3] + 0.1 * V[6])
        V_next[3] = dataModel.R[3] + gamma*(0.2 * V[4] + 0.8 * V[5])
        V_next[4] = dataModel.R[4] + gamma*(0.2 * V[1] + 0.5 * V[2] + 0.3 * V[3])
        V_next[5] = dataModel.R[5] + gamma*V[6]
        V_next[6] = dataModel.R[6]
        if np.allclose(V_next, V):  # 检查是否收敛
            break
    print("迭代次数 :", count)
    return V

def print_V(V):
    print(V)
    vv = np.around(V,3)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))

if __name__=="__main__":
    dataModel = dm.DataModel()
    gamma = 1

    V0 = linear_equations_iteration(dataModel, gamma)
    print_V(V0)

