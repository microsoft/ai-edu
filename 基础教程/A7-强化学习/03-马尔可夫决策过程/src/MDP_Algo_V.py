import numpy as np

def V_pi(States, Pi_sa, P_as, Rewards, gamma):
    num_state = 5
    V_curr = [0.0] * num_state
    V_next = [0.0] * num_state
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for curr_state in States:
            v_curr_sum = 0
            # 获得 状态->动作 策略概率
            next_actions_prob = Pi_sa[curr_state.value]
            # 遍历每个策略概率
            for action_value, action_prob in enumerate(next_actions_prob):
                # 获得 动作->状态 转移概率
                next_states_prob = P_as[action_value]
                v_sum = 0
                # 遍历每个转移概率
                for state_value, state_prob in enumerate(next_states_prob):
                    # math: \sum_{s'} P_{ss'}^a v_{\pi}(s')
                    v_sum += state_prob * V_next[state_value]
                #end for
                # math: \sum_a \pi(a|s) [R_s^a + \gamma \sum_{s'} P_{ss'}^a v_{\pi}(s')] 
                v_curr_sum += action_prob * (Rewards[action_value] + gamma * v_sum)
            # end for
            
            V_curr[curr_state.value] = v_curr_sum
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
