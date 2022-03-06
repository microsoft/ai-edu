import tqdm
import multiprocessing as mp
import math
import numpy as np

def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a - b))/b.shape[0])
    return err

def calculate_error(errors, episode, every_n_episode, V, ground_truth):
    if (episode % every_n_episode == 0):
        err = RMSE(V, ground_truth)
        errors.append(err)


# 一边生成序列一边计算
# 每一幕的中间状态都抛弃不用（造成浪费），只计算起始状态的V值
# 首次访问型，因为只对start_state计算V值，忽略中间的其他状态
def MC1(ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode):
    V = np.zeros((ds.num_states))
    errors = []
    # 多幕采样
    for episode in tqdm.trange(episodes):
        # 遍历每个状态(做为起始状态)
        for i in range(ds.num_states):
            start_state = ds.get_states()(i)
            # 终止状态，直接返回
            if (ds.is_end_state(start_state)):
                # 最后一个状态也可能有reward值
                V[start_state.value] += ds.get_reward(start_state)
                continue

            curr_state = start_state
            # math: g = r1 + \gamma*r2 + \gamma^2*r3 + ...
            g = 0
            power = 0

            while (True):
                if ds.is_end_state(curr_state):
                    break
                next_state, r = ds.step(curr_state)
                # math: \gamma^{t-1} * r
                g += math.pow(gamma, power) * r
                power += 1
                curr_state = next_state
            # end while
            V[start_state.value] += g
        # end for
        calculate_error(errors, episode+1, every_n_episode, V/(episode+1), ground_truth)
    # end for
    V = V / episodes
    return V, errors


# MC1的改进，反向计算G值，记录每个状态的G值，每次访问型
def MC2(ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode):
    V_value_count_pair = np.zeros((ds.num_states,2))  # state[total value, count of g]
    V_value_count_pair[:,1] = 1 # 避免被除数为0
    errors = []
    for episode in tqdm.trange(episodes):
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
        #endwhile
        num_step = len(trajectory)
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            state_value, reward = trajectory[t]
            g = gamma * g + reward
            V_value_count_pair[state_value, 0] += g     # total value
            V_value_count_pair[state_value, 1] += 1     # count
        #endfor
        calculate_error(errors, episode+1, every_n_episode, V_value_count_pair[:,0] / V_value_count_pair[:,1], ground_truth)
    #endfor
    V = V_value_count_pair[:,0] / V_value_count_pair[:,1]
    return V, errors


# 多状态同时更新的蒙特卡洛采样
# 注意输入V有初始状态
# constant-alpha
def MC3(ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode):
    V = np.zeros((ds.num_states))
    errors = []
    for episode in tqdm.trange(episodes):
        trajectory = []
        if (start_state is None):
            curr_state = ds.random_select_state()
        else:
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
        #endfor
        calculate_error(errors, episode+1, every_n_episode, V, ground_truth)
    #endfor
    return V, errors


# batch
def MC4(ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode):
    V = np.zeros((ds.num_states))
    G_value_count_pair = np.zeros((ds.num_states,2))  # state[total value, count of g]
    errors = []
    for episode in tqdm.trange(episodes):
        trajectory = []
        # randomly select on state as start state
        if (start_state is None):
            curr_state = ds.random_select_state()
        else:
            curr_state = start_state
        trajectory.append((curr_state.value, ds.get_reward(curr_state)))
        while True:
            # 到达终点，结束一幕，退出循环开始算分
            if (ds.is_end_state(curr_state)):
                break
            # 从环境获得下一个状态和奖励
            next_state, reward = ds.step(curr_state)
            #endif
            trajectory.append((next_state.value, reward))
            curr_state = next_state
        #endwhile
        # calculate G_t
        num_step = len(trajectory) 
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            state_value, reward = trajectory[t]
            g = gamma * g + reward
            G_value_count_pair[state_value, 0] += g     # total value
            G_value_count_pair[state_value, 1] += 1     # count
        #endfor

        if ((episode+1)%every_n_episode == 0):
            for state_value in range(ds.num_states):
                count = G_value_count_pair[state_value, 1]
                if (count == 0):
                    continue
                G = G_value_count_pair[state_value, 0] / count
                V[state_value] = V[state_value] + alpha * (G - V[state_value])
            G_value_count_pair[:,:] = 0
        #endwhile
        #endif
        calculate_error(errors, episode+1, every_n_episode, V, ground_truth)
    #endfor
    #print(update_count)
    return V, errors



def MC_all(ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode):
    V1 = np.zeros((ds.num_states))
    V2 = np.zeros((ds.num_states))
    V2_value_count_pair = np.zeros((ds.num_states,2))  # state[total value, count of g]
    V2_value_count_pair[:,1] = 1 # 避免被除数为0

    V3 = np.zeros((ds.num_states))    

    V4 = np.zeros((ds.num_states))
    V4_value_count_pair = np.zeros((ds.num_states,2))  # state[total value, count of g]

    errors1 = []
    errors2 = []
    errors3 = []
    errors4 = []

    for episode in tqdm.trange(episodes):
        trajectory = []
        start_state = ds.random_select_state()

        curr_state = start_state
        trajectory.append((curr_state.value, ds.get_reward(curr_state)))
        while True:
            # 到达终点，结束一幕，退出循环开始算分
            if (ds.is_end_state(curr_state)):
                break
            # 从环境获得下一个状态和奖励
            next_state, reward = ds.step(curr_state)
            trajectory.append((next_state.value, reward))
            curr_state = next_state
        #endwhile

        num_step = len(trajectory)
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            state_value, reward = trajectory[t]
            g = gamma * g + reward

            V2_value_count_pair[state_value, 0] += g     # total value
            V2_value_count_pair[state_value, 1] += 1     # count
            V3[state_value] = V3[state_value] + alpha * (g - V3[state_value])
            V4_value_count_pair[state_value, 0] += g     # total value
            V4_value_count_pair[state_value, 1] += 1     # count
        #endfor
        V1[start_state.value] += g

        if ((episode+1) % every_n_episode == 0):
            # V1
            calculate_error(errors1, episode+1, every_n_episode, V1*ds.num_states/(episode+1), ground_truth)
            # V2
            V2 = V2_value_count_pair[:,0] / V2_value_count_pair[:,1]
            calculate_error(errors2, episode+1, every_n_episode, V2, ground_truth)
            # V3
            calculate_error(errors3, episode+1, every_n_episode, V3, ground_truth)
            # V4
            for state_value in range(ds.num_states):
                count = V4_value_count_pair[state_value, 1]
                if (count == 0):
                    continue
                G = V4_value_count_pair[state_value, 0] / count
                V4[state_value] = V4[state_value] + alpha * (G - V4[state_value])
            #end for
            V4_value_count_pair[:,:] = 0

            calculate_error(errors4, episode+1, every_n_episode, V4, ground_truth)

    #endfor
    return V1*ds.num_states/episodes, errors1, V2, errors2, V3, errors3, V4, errors4
