import numpy as np
import CodeLifeCycle_DataModel as dm


# 原始迭代法
def raw_iteration_1(dataModel, gamma):
    print("---原始迭代法---")
    V = np.zeros(dataModel.N)
    V_old = V.copy()
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1
        V[0] = dataModel.R[0] + gamma*(0.7 * V[0] + 0.3 * V[1])
        V[1] = dataModel.R[1] + gamma*(0.6 * V[0] + 0.4 * V[2])
        V[2] = dataModel.R[2] + gamma*(0.9 * V[3] + 0.1 * V[6])
        V[3] = dataModel.R[3] + gamma*(0.2 * V[4] + 0.8 * V[5])
        V[4] = dataModel.R[4] + gamma*(0.2 * V[1] + 0.5 * V[2] + 0.3 * V[3])
        V[5] = dataModel.R[5] + gamma*V[6]
        V[6] = dataModel.R[6]
        if np.allclose(V_old, V):
            break
        V_old = V.copy()
    print("迭代次数 :", count)
    return V


# 矩阵迭代法
def matrix_iteration(dataModel, gamma):
    print("---矩阵迭代法---")
    V = np.zeros(dataModel.N)
    V_next = V.copy()
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1
        V_next = dataModel.R + gamma * np.dot(dataModel.P, V)
        if np.allclose(V_next, V):
            break
        V = V_next.copy()
    print("迭代次数 :", count)
    return V

# 矩阵迭代法
def matrix_iteration_singlearray(dataModel, gamma):
    print("---矩阵迭代法---")
#    V = dataModel.R
    V = np.zeros(dataModel.N)
    V_old = V.copy()
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1
        V = dataModel.R + gamma * np.dot(dataModel.P, V)
        if np.allclose(V_old, V):
            break
        V_old = V.copy()
    print("迭代次数 :", count)
    return V

# 贝尔曼方程双数组迭代
def Bellman_Iteration_DoubleArray(dataModel, gamma):
    print("---双数组迭代法---")
    V_curr = np.zeros(dataModel.N)
    V_next = np.zeros(dataModel.N)
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        # 遍历每一个 state 作为 start_state
        for curr_state in dataModel.S:
            # 得到转移概率
            next_states_probs = dataModel.P[curr_state.value]
            v_sum = 0
            # 计算下一个状态的 转移概率x状态值 的 和 v
            for next_state_value, next_state_prob in enumerate(next_states_probs):
                # if (prob[next_state] > 0.0):
                v_sum += next_state_prob * V_curr[next_state_value]
            # end for
            V_next[curr_state.value] = dataModel.R[curr_state.value] + gamma * v_sum
        # end for
        # 检查收敛性
        if np.allclose(V_next, V_curr):
            break
        # 把 V_curr 赋值给 V_next
        V_curr = V_next.copy()
        count += 1
    # end while
    print("迭代次数 :", count)
    return V_next

# 贝尔曼方程单数组就地更新
def Bellman_Iteration_SingleArray(dataModel, gamma):
    print("---单数组就地更新法---")
    V = np.zeros(dataModel.N)
    V_old = V.copy()
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1
        # 遍历每一个 state 作为 start_state
        for curr_state in dataModel.S:
            # 得到转移概率
            next_states_probs = dataModel.P[curr_state.value]
            v_sum = 0
            # 计算下一个状态的 转移概率x状态值 的 和 v
            for next_state_value, next_state_prob in enumerate(next_states_probs):
                # if (prob[next_state] > 0.0):
                v_sum += next_state_prob * V[next_state_value]
            # end for
            V[curr_state.value] = dataModel.R[curr_state.value] + gamma * v_sum
        # end for
        # 检查收敛性
        if np.allclose(V, V_old):
            break
        # 把 V_curr 赋值给 V_next
        V_old = V.copy()
    # end while
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

    V0 = matrix_iteration(dataModel, gamma)
    print_V(V0)

    V0 = matrix_iteration_singlearray(dataModel, gamma)
    print_V(V0)
