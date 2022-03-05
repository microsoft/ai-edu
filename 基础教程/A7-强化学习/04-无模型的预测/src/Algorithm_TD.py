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


def TD_0(ds, start_state, episodes, alpha, gamma, ground_truth, checkpoint, ):
    V = np.zeros((ds.num_states))
    errors = []
    for episode in tqdm.trange(episodes):
        if (start_state is None):
            curr_state = ds.random_select_state()
        else:
            curr_state = start_state
        #endif
        while True:
            # 到达终点，结束一幕
            if (ds.is_end_state(curr_state)):
                break
            next_state, reward = ds.step(curr_state)
            # math: V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]
            delta = reward + gamma * V[next_state.value] - V[curr_state.value]
            V[curr_state.value] = V[curr_state.value] + alpha * delta
            curr_state = next_state
            #endif
        #endwhile
        calculate_error(errors, episode, checkpoint, V, ground_truth)
    #endfor
    return V, errors

def TD_batch(ds, start_state, episodes, alpha, gamma, ground_truth, checkpoint):
    V = np.zeros((ds.num_states))
    errors = []
    batch_delta = np.zeros((ds.num_states, 2))
    for episode in tqdm.trange(episodes):
        if (start_state is None):
            curr_state = ds.random_select_state()
        else:
            curr_state = start_state
        #endif
        while True:
            # 到达终点，结束一幕
            if (ds.is_end_state(curr_state)):
                break
            # 随机游走
            next_state, reward = ds.step(curr_state)
            batch_delta[curr_state.value, 0] += reward + gamma * V[next_state.value] - V[curr_state.value]
            batch_delta[curr_state.value, 1] += 1
            curr_state = next_state
            #endif
        #endwhile
        if (episode+1) % checkpoint == 0:
            for state_value in range(ds.num_states):
                if (batch_delta[state_value, 1] > 0):
                    V[state_value] = V[state_value] + alpha * batch_delta[state_value, 0] / batch_delta[state_value, 1]
            batch_delta[:, :] = 0

        calculate_error(errors, episode, checkpoint, V, ground_truth)
    #endfor
    return V, errors
