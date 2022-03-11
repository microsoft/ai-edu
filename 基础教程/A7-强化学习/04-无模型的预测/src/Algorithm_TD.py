import tqdm
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

def Saras(env, from_start, episodes, ALPHA, GAMMA, ground_truth, checkpoint):
    def policy(Q, state, env):
        # 获得该状态下所有可能的action
        available_actions = env.get_actions(state)
        # e-贪心策略
        if np.random.rand() < 0.1:
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


    Q = np.zeros((env.state_space, env.action_space))
    rewards = np.zeros(episodes)
    for episode in tqdm.trange(episodes):
        curr_state = env.reset(from_start)
        actions = env.get_actions(curr_state)
        # put your policy here
        curr_action = np.random.choice(actions)
        is_done = False
        while not is_done:   # 到达终点，结束一幕
            prob, next_state, reward, is_done = env.step(curr_state, curr_action)
            next_action = policy(Q, next_state, env)
            # math: Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma Q(s',a') - Q(s,a)]
            delta = reward + GAMMA * Q[next_state, next_action] - Q[curr_state, curr_action]
            Q[curr_state, curr_action] += ALPHA * delta
            curr_state = next_state
            curr_action = next_action
            
            rewards[episode] += reward
        #endwhile
        #calculate_error(errors, episode, checkpoint, V, ground_truth)
    #endfor
    return Q, rewards

def Q_Learning(env, from_start, episodes, ALPHA, GAMMA, ground_truth, checkpoint):
    def policy(Q, state, env):
        # 获得该状态下所有可能的action
        available_actions = env.get_actions(state)
        if np.random.rand() < 0.1:
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


    Q = np.zeros((env.state_space, env.action_space))
    rewards = np.zeros(episodes)
    for episode in tqdm.trange(episodes):
        curr_state = env.reset(from_start)
        is_done = False
        while not is_done:   # 到达终点，结束一幕
            curr_action = policy(Q, curr_state, env)
            prob, next_state, reward, is_done = env.step(curr_state, curr_action)
            # math: Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_a Q(s',a) - Q(s,a)]
            available_actions = env.get_actions(next_state)
            delta = reward + GAMMA * np.max(Q[next_state][available_actions]) - Q[curr_state, curr_action]
            Q[curr_state, curr_action] += ALPHA * delta
            curr_state = next_state
            
            rewards[episode] += reward
        #endwhile
        #calculate_error(errors, episode, checkpoint, V, ground_truth)
    #endfor
    return Q, rewards

def draw_arrow(Q, width=6, height=3):
    np.set_printoptions(suppress=True)
    print(np.round(Q, 3))
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


import Data_FrozenLake3 as dfl3
import matplotlib.pyplot as plt

if __name__=="__main__":
    env = dfl3.Data_FrozenLake_Env()
    episodes = 200
    Q1,R1 = Saras(env, False, episodes, 0.01, 0.9, None, 10)
    Q2,R2 = Q_Learning(env, False, episodes, 0.01, 0.9, None, 10)
    print("Saras")
    draw_arrow(Q1)
    print("-"*20)
    print("Q-learning")
    draw_arrow(Q2)
    #plt.plot(R1, label="Saras")
    #plt.plot(R2, label="Q_ln")
    #plt.show()
