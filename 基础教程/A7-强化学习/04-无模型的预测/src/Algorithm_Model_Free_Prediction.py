import tqdm
import multiprocessing as mp
import math
import numpy as np

# 多状态同时更新的蒙特卡洛采样
def MC(V, ds, start_state, episodes, alpha, gamma):
    for i in tqdm.trange(episodes):
        trajectory = []
        curr_state = start_state
        trajectory.append((curr_state.value, ds.get_reward(curr_state)))
        while True:
            # 到达终点，结束一幕，退出循环开始算分
            if (ds.is_end_state(curr_state)):
                break
            # 左右随机游走
            next_state = ds.get_random_next_state(curr_state)
            #endif
            reward = ds.get_reward(next_state)
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


def TD(V, ds, start_state, episodes, alpha, gamma):
    for i in range(episodes):
        curr_state = start_state
        while True:
            # 到达终点，结束一幕
            if (ds.is_end_state(curr_state)):
                break
            # 随机游走
            next_state = ds.get_random_next_state(curr_state)
            #endif
            reward = ds.get_reward(next_state)
            # 立刻更新状态值，不等本幕结束
            # math: V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]
            V[curr_state.value] = V[curr_state.value] + alpha * (reward + gamma * V[next_state.value] - V[curr_state.value])
            curr_state = next_state
            #endif
        #endwhile
    #endfor
    return V


# 只针对输入的start_state一个状态做MC采样
def mc_single_process2(ds, start_state, episodes, gamma):
    sum_gain = 0
    curr_state = start_state
    for episode in tqdm.trange(episodes):
        if (ds.is_end_state(curr_state)):
            # 最后一个状态也可能有reward值
            return ds.get_reward(curr_state)
        g = ds.get_reward(curr_state)
        power = 1
        while (True):
            if ds.is_end_state(curr_state):
                break
            next_state = ds.get_random_next_state(curr_state)
            r = ds.get_reward(next_state)
            g += math.pow(gamma, power) * r
            power += 1
            curr_state = next_state
        # end while
        sum_gain += g
    # end for
    v = sum_gain / episodes
    return v  

def MonteCarol2(ds, gamma, episodes):
    pool = mp.Pool(processes=6)
    Vs = []
    results = []
    for start_state in ds.States:
        results.append(pool.apply_async(mc_single_process2, 
                                        args=(ds, start_state, episodes, gamma,)))
    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        Vs.append(v)

    return Vs



# 只针对输入的start_state一个状态做MC采样
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
        g = Rewards[curr_state_value]
        power = 1
        while (True):
            next_state_value = np.random.choice(num_state, p=TransMatrix[curr_state_value])
            r = Rewards[next_state_value]
            g += math.pow(gamma, power) * r
            if (States(next_state_value) in end_states):
                # 到达终点，分幕结束
                break
            else:
                power += 1
                curr_state_value = next_state_value
        # end while
        sum_gain += g
    # end for
    v = sum_gain / episodes
    return v  


# 单状态蒙特卡洛采样法
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
