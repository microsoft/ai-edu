import numpy as np

def V_pi(States, Pi_sa, Pr_as, Rewards, gamma):
    num_state = len(States)
    V_curr = [0.0] * num_state
    V_next = [0.0] * num_state
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for curr_state in States:
            v_sum = 0
            # 获得 状态->动作 策略概率
            next_actions_prob = Pi_sa[curr_state.value]
            # 遍历每个策略概率
            for action_value, action_prob in enumerate(next_actions_prob):
                # 获得 动作->状态 转移概率
                next_states_prob = Pr_as[action_value]
                q_sum = 0
                # 遍历每个转移概率
                for state_value, state_prob in enumerate(next_states_prob):
                    # math: \sum_{s'} P_{ss'}^a v_{\pi}(s')
                    q_sum += state_prob * V_next[state_value]
                #end for
                # math: \sum_a \pi(a|s) [R_s^a + \gamma \sum_{s'} P_{ss'}^a v_{\pi}(s')] 
                v_sum += action_prob * (Rewards[action_value] + gamma * q_sum)
            # end for
            
            V_curr[curr_state.value] = v_sum
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


def Q_pi(Actions, Pi_sa, Pr_as, Rewards, gamma):
    num_action = len(Actions)
    Q_curr = [0.0] * num_action
    Q_next = [0.0] * num_action
    count = 0
    # 迭代
    while (True):
        # 遍历每个action
        for curr_action in Actions:
            q_curr_sum = 0
            # 获得 动作->状态 转移概率
            next_states_prob = Pr_as[curr_action.value]
            # 遍历每个转移概率求和
            for state_value, state_prob in enumerate(next_states_prob):
                # 获得 状态->动作 策略概率
                next_actions_prob = Pi_sa[state_value]
                q_sum = 0
                # 遍历每个动作概率求和
                for action_value, action_prob in enumerate(next_actions_prob):
                    # math: \sum_{a'} \pi(a'|s')*q_{\pi}(s',a')
                    q_sum += action_prob * Q_next[action_value]
                #end for
                # math: \sum_{s'} P_{ss'}^a ( \sum_{a'} \pi(a'|s')q_{\pi}(s',a') )
                q_curr_sum += state_prob * q_sum
            # end for
            # math: q_{\pi}(s,a)=R_s^a + \sum_{s'} P_{ss'}^a ( \sum_{a'} \pi(a'|s')q_{\pi}(s',a') )
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


def Q_pi_from_V_pi(Actions, Pr_as, Rewards, gamma, v_pi):
    num_action = len(Actions)
    Q = [0.0] * num_action
    # 遍历每个action
    for curr_action in Actions:
        q_sum = 0
        # 获得 动作->状态 转移概率
        next_states_probs = Pr_as[curr_action.value]
        # 遍历每个转移概率求和
        for next_state_value, next_state_prob in enumerate(next_states_probs):
            # math: \sum_{s'} P_{ss'}^a v_{\pi}(s') 
            q_sum += next_state_prob * v_pi[next_state_value]
        # end for
        # math: q_{\pi}(s,a)=R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s')
        Q[curr_action.value] = Rewards[curr_action.value] + gamma * q_sum
    #endfor
    return Q
