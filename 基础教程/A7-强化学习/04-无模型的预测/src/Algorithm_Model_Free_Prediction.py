import tqdm
import multiprocessing as mp
import math
import numpy as np


# 多状态同时更新的蒙特卡洛采样
def MC_wrong(V, ds, start_state, episodes, alpha, gamma):
    for i in range(episodes):
        trajectory = []
        curr_state = start_state
        trajectory.append((curr_state.value, 0))
        while True:
            # 到达终点，结束一幕，退出循环开始算分
            if (ds.is_end_state(curr_state)):
                break
            # 左右随机游走
            next_state, reward = ds.step(curr_state)
            #endif
            trajectory.append((next_state.value, reward))
            curr_state = next_state
            #endif
        #endwhile
        # calculate G,V
        G = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for j in range(len(trajectory)-1, -1, -1):
            state_value, reward = trajectory[j]
            G = gamma * G + reward

        # 更新从状态开始到终止状态之前的所有V值
        for (state_value, reward) in trajectory[0:-1]:
            # math: V(s) \leftarrow V(s) + \alpha (G_t - V(s))
            # 这个实现有bug，G并不是G_t
            V[state_value] = V[state_value] + alpha * (G - V[state_value])
        
    #endfor
    return V


# 多状态同时更新的蒙特卡洛采样
def MC3(V, ds, start_state, episodes, alpha, gamma):
    for i in tqdm.trange(episodes):
        trajectory = []
        curr_state = start_state
        trajectory.append((curr_state.value, 0))
        while True:
            # 到达终点，结束一幕，退出循环开始算分
            if (ds.is_end_state(curr_state)):
                break
            # 从环境获得下一个状态和奖励
            next_state, reward = ds.step(curr_state)
            #endif
            trajectory.append((next_state.value, reward))
            curr_state = next_state

        # calculate G_t
        num_step = len(trajectory) 
        G = [0] * num_step
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            state_value, reward = trajectory[t]
            g = gamma * g + reward
            G[t] = g

        for t in range(num_step):
            state_value, reward = trajectory[t]
            V[state_value] = V[state_value] + alpha * (G[t] - V[state_value])
        
    #endfor
    return V


def TD(V, ds, start_state, episodes, alpha, gamma):
    for i in range(episodes):
        curr_state = start_state
        while True:
            # 到达终点，结束一幕
            if (ds.is_end_state(curr_state)):
                break
            # 随机游走
            next_state, reward = ds.step(curr_state)
            #endif
            # 立刻更新状态值，不等本幕结束
            # math: V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]
            V[curr_state.value] = V[curr_state.value] + alpha * (reward + gamma * V[next_state.value] - V[curr_state.value])
            curr_state = next_state
            #endif
        #endwhile
    #endfor
    return V


# 一边生成序列一边计算
# 每一幕的中间状态都抛弃不用（造成浪费），只计算起始状态的V值
# 首次访问型(只对start_state计算V值)
def MC1(ds, start_state, episodes, gamma):
    # 终止状态，直接返回
    if (ds.is_end_state(start_state)):
        # 最后一个状态也可能有reward值
        return ds.get_reward(start_state)

    # 对多个幕的结果求均值=期望E[G]
    sum_g = 0
    # 多幕采样
    for episode in tqdm.trange(episodes):
        curr_state = start_state
        # math: g = r1 + \gamma*r2 + \gamma^2*r3 + ...
        g = 0
        power = 0
        # 对每一幕
        while (True):
            if ds.is_end_state(curr_state):
                break
            next_state, r = ds.step(curr_state)
            # math: \gamma^{t-1} * r
            g += math.pow(gamma, power) * r
            power += 1
            curr_state = next_state
        # end while
        sum_g += g
    # end for
    v = sum_g / episodes
    return v  


def MonteCarol(ds, gamma, episodes):
    pool = mp.Pool(processes=4)
    Vs = []
    results = []
    for start_state in ds.get_states():
        results.append(
            pool.apply_async(MC1, args=(ds, start_state, episodes, gamma,)))

    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        Vs.append(v)

    return Vs


# MC1的改进，利用每个状态位
def MC2(V, ds, start_state, episodes, gamma):
    Vs = np.zeros((ds.num_states,2))  # state[total value, count of g]
    for i in tqdm.trange(episodes):
        trajectory = []
        curr_state = start_state
        trajectory.append((curr_state.value, 0))
        while True:
            # 到达终点，结束一幕，退出循环开始算分
            if (ds.is_end_state(curr_state)):
                break
            # 从环境获得下一个状态和奖励
            next_state, reward = ds.step(curr_state)
            #endif
            trajectory.append((next_state.value, reward))
            curr_state = next_state

        num_step = len(trajectory)
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            state_value, reward = trajectory[t]
            g = gamma * g + reward
            Vs[state_value, 0] += g     # total value
            Vs[state_value, 1] += 1     # count

    V = Vs[:,0] / Vs[:,1]
    #endfor
    return V
