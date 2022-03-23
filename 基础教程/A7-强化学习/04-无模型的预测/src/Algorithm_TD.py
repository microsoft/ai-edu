import tqdm
import numpy as np
import random

def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a - b))/b.shape[0])
    return err

def calculate_error(errors, episode, every_n_episode, V, ground_truth):
    if (episode % every_n_episode == 0):
        err = RMSE(V, ground_truth)
        errors.append(err)

def calculate_error_Q(env, errors, episode, checkpoint, Q):
    if ((episode+1) % checkpoint == 0):
        err = env.RMSE(Q)
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
        is_done  = False
        while True:
            # 到达终点，结束一幕
            if (ds.is_end_state(curr_state)):
                break
            next_state, reward, is_done = ds.step(curr_state)
            # math: V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]
            delta = reward + gamma * V[next_state.value] - V[curr_state.value]
            V[curr_state.value] = V[curr_state.value] + alpha * delta
            curr_state = next_state
            #endif
        #endwhile
        calculate_error(errors, episode, checkpoint, V, ground_truth)
    #endfor
    return V, errors


def update(V, states, rewards, alpha, gamma, update_t, n_step):
    G = 0
    for step in range(n_step):
        G += pow(gamma, step) * rewards[update_t + step + 1]
    # end for
    state_0 = states[update_t]
    state_n = states[update_t + n_step]
    G = G + pow(gamma, n_step) * V[state_n]
    V[state_0] += alpha * (G - V[state_0])

def update_tail(V, states, rewards, alpha, gamma, update_t, n_step):
    last_state = len(states) - 1
    while update_t < last_state:
        G = 0
        for step in range(last_state - update_t):
            G += pow(gamma, step) * rewards[update_t + step + 1]
        # end for
        state_0 = states[update_t]
        V[state_0] += alpha * (G - V[state_0])
        update_t += 1

def TD_n(ds, start_state, episodes, alpha, gamma, n_step, ground_truth, checkpoint):
    V = np.zeros(ds.num_states)
    errors = []
    for episode in tqdm.trange(episodes):
        curr_state = start_state
        states = [curr_state.value]
        rewards = [0]
        t = 0
        update_t = 0
        is_done = False
        while not is_done:
            t += 1
            next_state, reward, is_done = ds.step(curr_state)
            states.append(next_state.value)
            rewards.append(reward)
            curr_state = next_state
            if (t >= n_step):
                update(V, states, rewards, alpha, gamma, update_t, n_step)
                update_t += 1
            # end if
        # end while
        
        update_tail(V, states, rewards, alpha, gamma, update_t, n_step)

        calculate_error(errors, episode, checkpoint, V, ground_truth)
    # end for
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


# test
def test():
    Q = np.array([0,-1,0,-2,0,-1])
    actions = [1,3,5]
    q = Q[actions]
    print(q)
    b = np.argwhere(q == np.max(q))
    idx = np.random.choice(b.flatten())
    action = actions[idx]
    print(action)
    assert(action == 1 or action == 5)

def choose_action(Q, state, env, EPSILON):
    # 获得该状态下所有可能的action
    available_actions = env.get_actions(state)
    # e-贪心策略
    if np.random.rand() < EPSILON:
        action = np.random.choice(available_actions)
    else:
        # 取到与action对应的Q-values
        # e.g. Q[state]=[1,2,3,4]时，actions=[1,2,3], 则q_values=[2,3]
        q_values = Q[state][available_actions]
        # 得到q_values里的最大值的序号列表，有可能是多个
        # 如q_values=[-1,-2,-1], 则b=[0,2]
        b = np.argwhere(q_values == np.max(q_values))
        # 任意选一个,idx=0 or 2
        idx = np.random.choice(b.flatten())
        # 如果b=[0,2], 则action = 1 or 3
        action = available_actions[idx]
    return action

def choose_action_wrong(Q, state, env, EPSILON):
    # 获得该状态下所有可能的action
    available_actions = env.get_actions(state)
    # e-贪心策略
    if np.random.rand() < EPSILON:
        action = np.random.choice(available_actions)
    else:
        # 取到与action对应的Q-values
        # e.g. Q[state]=[1,2,3,4]时，actions=[1,2,3], 则q_values=[2,3]
        q_values = Q[state][available_actions]
        idx = np.argmax(q_values)
        action = available_actions[idx]
    return action


def Sarsa(env, from_start, episodes, ALPHA, GAMMA, EPSILON, checkpoint):
    Q = np.zeros((env.state_space, env.action_space))
    errors = []
    for episode in tqdm.trange(episodes):
        curr_state = env.reset(from_start)
        curr_action = choose_action(Q, curr_state, env, EPSILON)
        is_done = False
        while not is_done:   # 到达终点，结束一幕
            prob, next_state, reward, is_done = env.step(curr_state, curr_action)
            next_action = choose_action(Q, next_state, env, EPSILON)
            # math: Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma Q(s',a') - Q(s,a)]
            delta = reward + GAMMA * Q[next_state, next_action] - Q[curr_state, curr_action]
            Q[curr_state, curr_action] += ALPHA * delta
            curr_state = next_state
            curr_action = next_action
        #endwhile
    #endfor
    return Q, errors


def Q_Learning(env, from_start, episodes, ALPHA, GAMMA, EPSILON, checkpoint):
    Q = np.zeros((env.state_space, env.action_space))
    errors = []
    for episode in tqdm.trange(episodes):
        curr_state = env.reset(from_start)
        is_done = False
        while not is_done:   # 到达终点，结束一幕
            curr_action = choose_action(Q, curr_state, env, EPSILON)
            prob, next_state, reward, is_done = env.step(curr_state, curr_action)
            # math: Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_a Q(s',a) - Q(s,a)]
            available_actions = env.get_actions(next_state)
            # 因为有的动作缺失，比如边角状态不是具有上下左右四个动作
            # 缺失的动作的Q值被初始化为0，且保持不变
            # 其它具备的动作的Q值很可能是负值（因为R为负)，所0成为max，要避免这种情况
            delta = reward + GAMMA * np.max(Q[next_state][available_actions]) - Q[curr_state, curr_action]
            Q[curr_state, curr_action] += ALPHA * delta
            curr_state = next_state
        #endwhile
        #calculate_error_Q(env, errors, episode, checkpoint, Q)
    #endfor
    return Q, errors


def E_Sarsa(env, from_start, episodes, ALPHA, GAMMA, EPSILON, checkpoint):

    def E_pi():
        available_actions = env.get_actions(next_state)
        q_actions = Q[next_state][available_actions]
        return np.sum(q_actions) / len(available_actions)
        best_action = np.argmax(q_actions)
        q_expected = (1 - EPSILON) * Q[next_state, best_action] + \
                     EPSILON * np.sum(q_actions) / len(available_actions)
        return q_expected

    Q = np.zeros((env.state_space, env.action_space))
    steps = []
    for episode in tqdm.trange(episodes):
        curr_state = env.reset(from_start)
        is_done = False
        while not is_done:   # 到达终点，结束一幕
            curr_action = choose_action(Q, curr_state, env, EPSILON)
            prob, next_state, reward, is_done = env.step(curr_state, curr_action)
            # math: Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \sum \pi(a|s')*Q(s',a) - Q(s,a)]
            q_expected = E_pi()
            Q[curr_state, curr_action] += ALPHA * (reward + GAMMA * q_expected - Q[curr_state, curr_action])
            curr_state = next_state
        #endwhile
        #calculate_error_Q(env, errors, episode, checkpoint, Q)
    #endfor
    return Q, steps


def Double_Q(env, from_start, episodes, ALPHA, GAMMA, EPSILON, checkpoint):
    Q1 = np.zeros((env.state_space, env.action_space))
    Q2 = np.zeros((env.state_space, env.action_space))
    errors = []
    for episode in tqdm.trange(episodes):
        curr_state = env.reset(from_start)
        is_done = False
        while not is_done:   # 到达终点，结束一幕
            Q = Q1 + Q2
            curr_action = choose_action(Q, curr_state, env, EPSILON)
            prob, next_state, reward, is_done = env.step(curr_state, curr_action)
            available_actions = env.get_actions(next_state)
            if (random.random() > 0.5):
                action_id = np.argmax(Q1[next_state][available_actions])
                q1_max_action = available_actions[action_id]
                Q1[curr_state, curr_action] += ALPHA * (reward + GAMMA * Q2[next_state, q1_max_action] - Q1[curr_state, curr_action])
            else:
                action_id = np.argmax(Q2[next_state][available_actions])
                q2_max_action = available_actions[action_id]
                Q2[curr_state, curr_action] += ALPHA * (reward + GAMMA * Q1[next_state, q2_max_action] - Q2[curr_state, curr_action])
            #endif
            curr_state = next_state
        #endwhile
        #calculate_error_Q(env, errors, episode, checkpoint, Q)
    #endfor
    return Q, errors



def draw_arrow(Q, width=6):
    np.set_printoptions(suppress=True)
    #print(np.round(Q, 3))
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if (Q[i,j] == 0):
                Q[i,j]=-10

    chars = [0x2191, 0x2192, 0x2193, 0x2190]
    for i in range(Q.shape[0]):
        if np.sum(Q[i,:]) == -40:
            print("O", end="")
        else:
            idx = np.argmax(Q[i,:])
            print(chr(chars[idx]), end="")
        print(" ", end="")
        if ((i+1) % width == 0):
            print("")


import Data_FrozenLake2 as dfl2
import Data_FrozenLake as dfl
import Data_CliffWalking as dc
import matplotlib.pyplot as plt

import Algorithm_MRP as algoMRP

def FrozenLake_Matrix(gamma):

    num_state = ds.get_TransMatrix().shape[0]
    I = np.eye(num_state)
    tmp1 = I - gamma * ds.get_TransMatrix()
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, ds.get_rewards())
    return vs


if __name__=="__main__":
    episodes = 10000
    EPSILON = 0.1
    GAMMA = 0.9
    ALPHA = 0.02
    n_step = 5
    ds = dfl.Data_Frozen_Lake()
    ground_truth = FrozenLake_Matrix(GAMMA)
    V1, E1 = TD_0(ds, ds.get_start_state(), episodes, ALPHA, GAMMA, ground_truth, 10)
    print(np.array(V1).reshape(4,4))
    V2, E2 = TD_n(ds, ds.get_start_state(), episodes, ALPHA, GAMMA, n_step, ground_truth, 10)
    print(V2.reshape(4,4))

    plt.plot(E1, label="TD-0")
    plt.plot(E2, label="TD-" + str(n_step))
    plt.legend()
    plt.show()
