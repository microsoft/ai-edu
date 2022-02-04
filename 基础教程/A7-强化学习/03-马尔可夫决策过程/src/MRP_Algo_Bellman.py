import numpy as np

def run(States, TransMatrix, Rewards, gamma):
    num_states = len(Rewards)
    V_curr = [0.0] * num_states
    V_next = [0.0] * num_states
    count = 0
    while (count < 1000):
        # 遍历每一个 state 作为 start_state
        for start_state in States:
            # 得到转移概率
            next_states_probs = TransMatrix[start_state.value]
            v_sum = 0
            # 计算下一个状态的 转移概率*状态值 的 和 v
            for next_state_value, next_state_prob in enumerate(next_states_probs):
                # if (prob[next_state] > 0.0):
                v_sum += next_state_prob * V_next[next_state_value]
            # end for
            V_curr[start_state.value] = Rewards[start_state.value] + gamma * v_sum
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
