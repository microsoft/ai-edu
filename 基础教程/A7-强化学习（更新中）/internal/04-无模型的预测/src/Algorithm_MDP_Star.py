import numpy as np

# state value function
# Pi_sa: 策略函数，用于选择动作
# P_as: 状态转移概率，执行动作后，到达下个状态的概率
def V_star(env, gamma):
    V_curr = np.zeros(env.state_space)
    V_next = np.zeros(env.state_space)
    count = 0
    # 迭代
    while (count < 100):
        # 遍历所有状态 s
        for curr_state in env.States:
            if (curr_state in env.EndStates):
                continue
            Actions = env.get_actions(curr_state.value)
            list_v = []
            for curr_action in Actions:
                states_trans = env.P[curr_state.value][curr_action]
                sum_v = 0
                for (prob, next_state, reward, is_end) in states_trans:
                    sum_v += prob * (reward + gamma * V_curr[next_state])
                #endfor
                list_v.append(sum_v)
            #endfor
            V_curr[curr_state.value] = max(list_v)
        #endfor
        # 检查收敛性
        if np.allclose(V_next, V_curr):
            break
        # 把 V_curr 赋值给 V_next
        V_next = V_curr.copy()
        count += 1
    # end while
    print(count)
    return V_next

# action value function
def Q_star(env, gamma):
    Q_curr = np.zeros((env.state_space, env.action_space))
    Q_next = np.zeros((env.state_space, env.action_space))
    count = 0
    # 迭代
    while (count < 100):
        # 遍历每个action
        for curr_state in env.States:
            Actions = env.get_actions(curr_state.value)
            for curr_action in Actions:
                states_trans = env.P[curr_state.value][curr_action]
                sum_q = 0
                for (prob, next_state, reward, is_end) in states_trans:
                    available_ctions = env.get_actions(next_state)
                    max_q = np.max(Q_curr[next_state, available_ctions])
                    sum_q += prob * (reward + gamma * max_q)
                #endfor
                Q_curr[curr_state.value, curr_action] = sum_q
            #endfor
        # 检查收敛性
        if np.allclose(Q_next, Q_curr):
            break
        # 把 Q_curr 赋值给 Q_next
        Q_next = Q_curr.copy()
        count += 1
    # end while
    print(count)
    return Q_next



def draw_arrow(Q, width=4):
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
import Data_CliffWalking as dcw

env = dfl2.Env()
#env = dcw.Env()
gamma = 0.9
Q = Q_star(env, gamma)
print("Q*")
print(Q)
draw_arrow(Q, width=4)

V = V_star(env, gamma)
print("V*")
print(V.reshape(4,4))
