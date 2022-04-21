import numpy as np
import CodeLifeCycle_DataModel_P as dm

# 单数组原始迭代法
def linear_equations_iteration_single_array(dataModel, gamma):
    print("---单数组原始迭代法---")
    V = np.zeros(dataModel.N)   # 初始化为全 0
    count = 0   # 迭代计数器
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1          # 计数器+1
        V_old = V.copy()    # 备份上一次的迭代值用于检查收敛性
        # 线性方程组
        V[0] = dataModel.R[0] + gamma*(0.7 * V[0] + 0.3 * V[1])
        V[1] = dataModel.R[1] + gamma*(0.6 * V[0] + 0.4 * V[2])
        V[2] = dataModel.R[2] + gamma*(0.9 * V[3] + 0.1 * V[6])
        V[3] = dataModel.R[3] + gamma*(0.2 * V[4] + 0.8 * V[5])
        V[4] = dataModel.R[4] + gamma*(0.2 * V[1] + 0.5 * V[2] + 0.3 * V[3])
        V[5] = dataModel.R[5] + gamma*(1.0 * V[6])
        V[6] = dataModel.R[6]
        if np.allclose(V_old, V):   # 检查收敛
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

    V = linear_equations_iteration_single_array(dataModel, gamma)
    print_V(V)