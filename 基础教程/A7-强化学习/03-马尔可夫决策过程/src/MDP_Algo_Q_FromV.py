
def Q2_pi(Actions, P_as, Rewards, gamma, v):
    num_action = 8
    Q = [0.0] * num_action
    # 遍历每个action
    for curr_action in Actions:
        q_sum = 0
        # 获得 动作->状态 转移概率
        next_states_probs = P_as[curr_action.value]
        # 遍历每个转移概率求和
        for next_state_value, next_state_prob in enumerate(next_states_probs):
            # math: \sum_{s'} P_{ss'}^a v_{\pi}(s') 
            q_sum += next_state_prob * v[next_state_value]
        # end for
        # math: q_{\pi}(s,a)=R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s')
        Q[curr_action.value] = Rewards[curr_action.value] + gamma * q_sum
    #endfor
    return Q
