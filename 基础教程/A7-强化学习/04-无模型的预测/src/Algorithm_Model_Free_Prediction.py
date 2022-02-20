import tqdm
import multiprocessing as mp
import math
import numpy as np

# 多状态同时更新的蒙特卡洛采样
def MC(V, ds, start_state, episodes, alpha, gamma):
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
        
        """
        # 只更新起始状态的V值,中间的都忽略,但是需要遍历所有状态为start_state
        s,r = trajectory[0]
        V[s] = V[s] + alpha * (G - V[s])
        """

        # 更新从状态开始到终止状态之前的所有V值
        for (state_value, reward) in trajectory[0:-1]:
            # math: V(s) \leftarrow V(s) + \alpha (G - V(s))
            V[state_value] = V[state_value] + alpha * (G - V[state_value])
        
    #endfor
    return V


# 多状态同时更新的蒙特卡洛采样
def MC2(V, ds, start_state, episodes, alpha, gamma):
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

        num_step = len(trajectory) 
        G = [0] * num_step
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for j in range(num_step-1, -1, -1):
            state_value, reward = trajectory[j]
            g = gamma * g + reward
            G[j] = g
        
        """
        # 只更新起始状态的V值,中间的都忽略,但是需要遍历所有状态为start_state
        s,r = trajectory[0]
        V[s] = V[s] + alpha * (G - V[s])
        """

        for j in range(num_step):
            state_value, reward = trajectory[j]
            V[state_value] = V[state_value] + alpha * (G[j] - V[state_value])
        
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


# 只针对输入的start_state一个状态做MC采样
def mc_single_process(ds, start_state, episodes, gamma):
    # 终止状态，直接返回
    if (ds.is_end_state(start_state)):
        # 最后一个状态也可能有reward值
        return ds.get_reward(start_state)

    # 对多个幕的结果求均值=期望E[G]
    sum_g = 0

    for episode in tqdm.trange(episodes):
        curr_state = start_state
        # g = r1 + gamma*r2 + gamma^2*r3 + gamma^3*r4
        g = 0
        power = 0
        # 对每一幕
        while (True):
            if ds.is_end_state(curr_state):
                break
            next_state, r = ds.step(curr_state)
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
            pool.apply_async(mc_single_process, args=(ds, start_state, episodes, gamma,)))

    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        Vs.append(v)

    return Vs
