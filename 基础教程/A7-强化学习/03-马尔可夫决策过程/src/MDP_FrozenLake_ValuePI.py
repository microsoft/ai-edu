import Data_FrozenLake2 as dfl2

import numpy as np

def V_pi(States, dataParser, gamma):
    num_state = 16
    V_curr = [0.0] * num_state
    V_next = [0.0] * num_state
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for curr_state in States:
            v_curr_sum = 0
            # 获得 状态->动作 策略概率
            actions_data = dataParser.get_next_actions(curr_state)
            # 遍历每个策略概率
            for action_data in actions_data:
                next_action_value, next_action_prob, reward = dataParser.get_action_pi_reward(action_data)
                # 获得 动作->状态 转移概率
                next_states_probs = dataParser.get_action_states_probs(action_data)
                #next_states_prob = P_as[action_value]
                v_sum = 0
                # 遍历每个转移概率
                for [next_state_value, next_state_prob] in next_states_probs:
                    # math: \sum_{s'} P_{ss'}^a v_{\pi}(s')
                    v_sum += next_state_prob * V_next[next_state_value]
                #end for
                # math: \sum_a \pi(a|s) [R_s^a + \gamma \sum_{s'} P_{ss'}^a v_{\pi}(s')] 
                v_curr_sum += next_action_prob * (reward + gamma * v_sum)
            # end for
            V_curr[curr_state.value] = v_curr_sum
        #endfor
        # 检查收敛性
        if np.allclose(V_next, V_curr):
            break
        # 把 V_curr 赋值给 V_next 迭代
        V_next = V_curr.copy()
        count += 1
    # end while
    print(count)
    return V_next

def Q2_pi(Actions, dataParser, gamma, vs):
    Q = {}
    # 遍历每个action
    for curr_action in Actions:
        q_sum = 0
        # 获得 动作->状态 转移概率
        reward, next_states_probs = dataParser.get_next_states_probs(curr_action.value)
        if (reward is None):
            continue
        #next_states_probs = P_as[curr_action.value]
        # 遍历每个转移概率求和
        for [next_state_value, next_state_prob] in next_states_probs:
            # math: \sum_{s'} P_{ss'}^a v_{\pi}(s') 
            q_sum += next_state_prob * vs[next_state_value]
        # end for
        # math: q_{\pi}(s,a)=R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s')
        q = reward + gamma * q_sum
        Q[curr_action.name] = q
    #endfor
    return Q


if __name__=="__main__":
    gamma = 0.9
    dataParser = dfl2.DataParser()
    vs = V_pi(dfl2.States, dataParser, gamma)
    print(np.round(np.array(vs).reshape(4,4), 2))
    Q = Q2_pi(dfl2.Actions, dataParser, gamma, vs)
    for q in Q:
        print(q, "={:.4f}".format(Q[q]))
