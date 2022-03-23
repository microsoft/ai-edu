import numpy as np

# state value function
# Pi_sa: 策略函数，用于选择动作
# P_as: 状态转移概率，执行动作后，到达下个状态的概率
def V_star(States, Pi_sa, P_as, Rewards, gamma):
    num_state = len(States)
    V_curr = [0.0] * num_state
    V_next = [0.0] * num_state
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for curr_state in States:
            list_v = []
            # 获得 状态->动作 策略概率
            next_actions_probs = Pi_sa[curr_state.value]
            # 遍历每个策略概率
            for action_value, action_prob in enumerate(next_actions_probs):
                if (action_prob > 0.0):
                    # 获得 动作->状态 转移概率
                    next_states_probs = P_as[action_value]
                    v_sum = 0
                    # 遍历每个转移概率
                    for state_value, state_prob in enumerate(next_states_probs):
                        # math: \sum_{s'} P_{ss'}^a v_{\pi}(s')
                        v_sum += state_prob * V_next[state_value]
                    #end for
                    # math: \max [R_s^a + \gamma \sum_{s'} P_{ss'}^a v_{\pi}(s')] 
                    list_v.append(Rewards[action_value] + gamma * v_sum)
            # end for
            if (len(list_v) > 0):
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
def Q_star(Actions, Pi_sa, P_as, Rewards, gamma):
    num_action = len(Actions)
    Q_curr = [0.0] * num_action
    Q_next = [0.0] * num_action
    count = 0
    # 迭代
    while (count < 100):
        # 遍历每个action
        for curr_action in Actions:
            q_curr_sum = 0
            if (curr_action == Actions.Sleep):
                continue
            # 获得 动作->状态 转移概率
            next_states_probs = P_as[curr_action.value]
            # 遍历每个转移概率求和
            for state_value, state_prob in enumerate(next_states_probs):
                # 获得 状态->动作 策略概率
                next_actions_probs = Pi_sa[state_value]
                list_q = []
                # 遍历每个动作概率求和
                for next_action_value, next_action_prob in enumerate(next_actions_probs):
                    if (next_action_prob > 0.0):
                        # math: q_{\pi}(s',a')
                        list_q.append(Q_next[next_action_value])
                #end for
                # math: \sum_{a'} P_{ss'}^a \max q_{\pi}(s',a') 
                if (len(list_q) > 0):
                    q_curr_sum += state_prob * max(list_q) 
            # end for
            # math: R_s^a + \gamma  ( \sum_{a'} P_{ss'}^a \max q_{\pi}(s',a') )
            Q_curr[curr_action.value] = Rewards[curr_action.value] + gamma * q_curr_sum
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


# math: q_*(s,a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s')
def Q_star_from_V_star(Actions, P_as, Rewards, gamma, v_star):
    num_action = len(Actions)
    Q = [0.0] * num_action
    # 遍历每个action
    for curr_action in Actions:
        q_sum = 0
        if (curr_action == Actions.Sleep):
            continue
        # 获得 动作->状态 转移概率
        next_states_probs = P_as[curr_action.value]
        # 遍历每个转移概率求和
        for next_state_value, next_state_prob in enumerate(next_states_probs):
            # math: \sum_{a'} P_{ss'}^a v_{*}(s') 
            q_sum += next_state_prob * v_star[next_state_value]
        # end for
        # math: R_s^a + \gamma  ( \sum_{a'} P_{ss'}^a \max q_{\pi}(s',a') )
        Q[curr_action.value] = Rewards[curr_action.value] + gamma * q_sum
    #endfor
    return Q
