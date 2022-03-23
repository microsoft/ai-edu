import numpy as np
import Data_Random_Walker as ds
import matplotlib.pyplot as plt
import Algorithm_MRP as algoMRP



def RMSE(V):
    return np.sqrt(np.sum(np.square(V - ground_truth[1:6]))/V.shape[0])


# 多状态同时更新的蒙特卡洛采样
def MC(ds, start_state, episodes, alpha, gamma):
    V = np.zeros(7)
    V[1:6] = 0.5
    errors = []
    for i in range(episodes):
        errors.append(RMSE(V[1:6]))
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
            # math: V(s) \leftarrow V(s) + \alpha (G - V(s))
            V[state_value] = V[state_value] + alpha * (G - V[state_value])
        
    #endfor
    return errors


# 多状态同时更新的蒙特卡洛采样
def MC2(ds, start_state, episodes, alpha, gamma):
    V = np.zeros(7)
    V[1:6] = 0.5

    errors = []
    for i in range(episodes):
        errors.append(RMSE(V[1:6]))

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
        
        for j in range(num_step):
            state_value, reward = trajectory[j]
            V[state_value] = V[state_value] + alpha * (G[j] - V[state_value])
        
    #endfor
    return errors


def TD(ds, start_state, episodes, alpha, gamma):
    V = np.zeros(7)
    V[1:6] = 0.5
    errors = []
    for i in range(episodes):
        errors.append(RMSE(V[1:6]))
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
    return errors

import tqdm

def Runs(run_num, episode, alpha, gamma, func):
    average_err = np.zeros(episode)
    for i in tqdm.trange(run_num):
        errors = func(ds.Data_Random_Walker(), ds.States.RoadC, episode, alpha, gamma)
        average_err += np.asarray(errors)
    #endfor
    average_err /= run_num
    return average_err

if __name__=="__main__":
    gamma = 0.9

    global ground_truth
    ground_truth = algoMRP.Matrix(ds, gamma)
    print(ground_truth)

    Errors = []
    episode = 100
    run_num = 100

    #alphas_mc = [0.01, 0.02, 0.03, 0.04]
    alphas_mc = [0.01, 0.02, 0.03]
    for alpha in alphas_mc:
        errors = Runs(run_num, episode, alpha, gamma, MC)
        Errors.append(errors)

    for alpha in alphas_mc:
        errors = Runs(run_num, episode, alpha, gamma, MC2)
        Errors.append(errors)

    
    alphas_td = [0.05, 0.1, 0.15]
    for alpha in alphas_td:
        errors = Runs(run_num, episode, alpha, gamma, TD)
        Errors.append(errors)
    

    alphas = alphas_mc + alphas_mc + alphas_td
    i = 0
    for errors in Errors:
        if i <= 2:
            plt.plot(errors, label=str(alphas[i]))
        elif i <= 5:
            plt.plot(errors, "--", label=str(alphas[i]))
        else:
            plt.plot(errors, ".-", label=str(alphas[i]))
        i += 1
    
    plt.grid()
    plt.legend()
    plt.show()

