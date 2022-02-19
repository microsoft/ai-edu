import math
import numpy as np
import tqdm
import multiprocessing as mp

def mc_single_process(
    Rewards, TransMatrix, States, 
    start_state, end_states, episodes, gamma):
    num_state = len(Rewards)
    sum_gain = 0
    for episode in tqdm.trange(episodes):
        if (start_state in end_states):
            # 最后一个状态也可能有reward值
            return Rewards[start_state.value]
        curr_state_value = start_state.value
        gain = Rewards[curr_state_value]
        power = 1
        while (True):
            next_state_value = np.random.choice(
                num_state, p=TransMatrix[curr_state_value])
            r = Rewards[next_state_value]
            gain += math.pow(gamma, power) * r
            if (States(next_state_value) in end_states):
                # 到达终点，分幕结束
                break
            else:
                power += 1
                curr_state_value = next_state_value
        # end while
        sum_gain += gain
    # end for
    v = sum_gain / episodes
    return v  

# 蒙特卡洛采样法
def MonteCarol(ds, end_states, gamma, episodes):
    pool = mp.Pool(processes=6)
    Vs = []
    results = []
    for start_state in ds.States:
        results.append(pool.apply_async(mc_single_process, 
        args=(ds.Rewards, ds.TransMatrix, ds.States, start_state, end_states, episodes, gamma,)))
    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        Vs.append(v)

    return Vs

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
    V_curr = [0.0] * num_states
    V_next = [0.0] * num_states
    count = 0
    while (count < 1000):
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
