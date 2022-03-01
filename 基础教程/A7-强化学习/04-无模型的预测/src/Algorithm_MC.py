import tqdm
import multiprocessing as mp
import math
import numpy as np

'''
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
'''


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


# 一边生成序列一边计算
# 每一幕的中间状态都抛弃不用（造成浪费），只计算起始状态的V值
# 首次访问型(只对start_state计算V值)
def MC1(ds, episodes, gamma):
    V = np.zeros(ds.num_states)
    # 遍历每个状态(做为起始状态)
    for i in tqdm.trange(ds.num_states):
        start_state = ds.get_states()(i)
        # 终止状态，直接返回
        if (ds.is_end_state(start_state)):
            # 最后一个状态也可能有reward值
            V[start_state.value] = ds.get_reward(start_state)
            continue

        # 对多个幕的结果求均值=期望E[G]
        sum_g = 0
        # 多幕采样
        for episode in range(episodes):
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
        V[start_state.value] = sum_g / episodes
    return V

# MC1的改进，利用每个状态位
def MC2(ds, start_state, episodes, gamma):
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


# 多状态同时更新的蒙特卡洛采样
# 注意输入V有初始状态
def MC3(V, ds, start_state, episodes, alpha, gamma):
    for i in range(episodes):
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
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            state_value, reward = trajectory[t]
            g = gamma * g + reward
            #G[t] = g
            V[state_value] = V[state_value] + alpha * (g - V[state_value])

    return V


# batch
def MC4(V, ds, start_state, episodes, alpha, gamma):
    trajectory_G = []
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
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            state_value, reward = trajectory[t]
            g = gamma * g + reward
            trajectory_G.append((state_value, g))
            

        
        # batch update
        while True:
            V_old = V.copy()
            updates = np.zeros(16)
            count = 0
            for (state, G) in trajectory_G:
                updates[state] += G - V[state]
                count += 1
                if (count == 1000):
                    #print(updates[state])
            # 在这里 乘以 alpha 是为了节省计算量，本来应该在上面一行中 乘以 alpha
                    updates = updates / count * alpha
                    # 一次性更新所有V值
                    V += updates
                    updates = np.zeros(16)
                    count = 0
                #end if
            #end for
            if (count > 0):
                updates = updates / count * alpha
                # 一次性更新所有V值
                V += updates
            if (np.allclose(V_old, V, alpha)):
                break   # converage
            

    return V
