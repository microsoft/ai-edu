import numpy as np
import CodeLifeCycle_DataModel_D as dm


# 贝尔曼方程单数组就地更新
def Bellman_iteration_single_array(dataModel, gamma):
    print("---单数组就地更新法---")
    V = np.zeros(dataModel.N)
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1
        V_old = V.copy()
        # 遍历每一个 state 作为 curr_state
        for curr_state in dataModel.S:
            # 得到转移概率
            list_state_prob  = dataModel.get_next(curr_state)
            # 计算 \sum(P·V)
            v_sum = 0
            for next_state, next_prob in list_state_prob:
                v_sum += next_prob * V[next_state.value]
            # 计算 V = R + gamma * \sum(P·V)
            V[curr_state.value] = dataModel.R[curr_state.value] + gamma * v_sum
        # 检查收敛性
        if np.allclose(V, V_old):
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

    V = Bellman_iteration_single_array(dataModel, gamma)
    print_V(V)