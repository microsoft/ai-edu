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


def TD_0(ds, start_state, episodes, alpha, gamma, ground_truth, checkpoint):
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

def TD_batch(ds, start_state, episodes, alpha, gamma, ground_truth, checkpoint, batch_num=0):
    V = np.zeros((ds.num_states))
    errors = []
    if batch_num == 0:
        batch_num = checkpoint
    batch_delta = np.zeros((ds.num_states, 2))
    batch_counter = 0

    for episode in tqdm.trange(episodes):
        batch_counter += 1
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
            batch_delta[curr_state.value, 0] += \
                reward + gamma * V[next_state.value] - V[curr_state.value]
            batch_delta[curr_state.value, 1] += 1
            curr_state = next_state
            #endif
        #endwhile

        if (batch_counter % batch_num == 0):
            for state_value in range(ds.num_states):
                count = batch_delta[state_value, 1]
                if (count > 0):
                    V[state_value] += \
                        alpha * batch_delta[state_value, 0] / count
            batch_delta[:, :] = 0

        calculate_error(errors, episode, checkpoint, V, ground_truth)
    #endfor
    return V, errors


def Saras(env, from_start, episodes, alpha, gamma, ground_truth, checkpoint):
    def policy(Q, state, env):
        actions = env.get_actions(state)
        if np.random.rand() < 0.1:
            action = np.random.choice(actions)
        else:
            q_values = Q[state]
            action = np.random.choice([a for a,v in enumerate(q_values) if (v == np.max(q_values))])
        return action


    Q = np.zeros((env.state_space, env.action_space))
    errors = []

    for episode in range(episodes):
        #print("-----------------")
        curr_state = env.reset(from_start)
        actions = env.get_actions(curr_state)
        # put your policy here
        curr_action = np.random.choice(actions)
        is_done = False
        tmp = []
        
        while not is_done:   # 到达终点，结束一幕
            tmp.append(curr_state)
            prob, next_state, reward, is_done = env.step(curr_state, curr_action)
            #print(curr_state, next_state, reward)
            next_action = policy(Q, next_state, env)
            # math: Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma Q(s',a') - Q(s,a)]
            delta = reward + gamma * Q[next_state, next_action] - Q[curr_state, curr_action]
            Q[curr_state, curr_action] += alpha * delta
            curr_state = next_state
            curr_action = next_action
            #endif
        #endwhile
        #calculate_error(errors, episode, checkpoint, V, ground_truth)
    #endfor
    return Q

import Data_FrozenLake2 as dfl2

if __name__=="__main__":
    env = dfl2.Data_FrozenLake_Env()
    Q = Saras(env, True, 1000, 0.01, 0.9, None, 10)
    print(Q)
