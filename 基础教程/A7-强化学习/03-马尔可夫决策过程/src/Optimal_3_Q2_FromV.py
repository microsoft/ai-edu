from enum import Enum
import numpy as np
from Optimal_1_V_StateValue import *

# math: q_*(s,a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s')

def Q2_star(gamma, v_star):
    num_action = 8
    num_state = 5
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

if __name__=="__main__":
    gamma = 1
    v_star = V_star(gamma)
    print(v_star)
    v = Q2_star(gamma, v_star)
    for action in Actions:
        print(action, "= {:.1f}".format(v[action.value]))
