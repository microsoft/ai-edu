from enum import Enum
import numpy as np
from MDP_0_Base import *

def Q_star(gamma):
    num_action = 8
    num_state = 5
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

if __name__=="__main__":
    print(Rewards)
    gamma = 1
    v = Q_star(gamma)
    for action in Actions:
        print(action, "= {:.1f}".format(v[action.value]))
