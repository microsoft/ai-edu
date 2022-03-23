import numpy as np

# 矩阵法
def Matrix(ds, gamma):
    num_state = ds.TransMatrix.shape[0]
    I = np.eye(num_state)
    tmp1 = I - gamma * ds.TransMatrix
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, ds.Rewards)

    return vs

# 贝尔曼方程迭代
def Bellman(ds, gamma):
    num_states = len(ds.Rewards)
    V_curr = np.zeros(num_states)
    V_next = np.zeros(num_states)
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        # 遍历每一个 state 作为 start_state
        for start_state in ds.States:
            # 得到转移概率
            next_states_probs = ds.TransMatrix[start_state.value]
            v_sum = 0
            # 计算下一个状态的 转移概率*状态值 的 和 v
            for next_state_value, next_state_prob in enumerate(next_states_probs):
                # if (prob[next_state] > 0.0):
                v_sum += next_state_prob * V_next[next_state_value]
            # end for
            V_curr[start_state.value] = ds.Rewards[start_state.value] + gamma * v_sum
        # end for
        # 检查收敛性
        if np.allclose(V_next, V_curr):
            break
        # 把 V_curr 赋值给 V_next
        V_next = V_curr.copy()
        count += 1
    # end while
    print(count)
    return V_next
