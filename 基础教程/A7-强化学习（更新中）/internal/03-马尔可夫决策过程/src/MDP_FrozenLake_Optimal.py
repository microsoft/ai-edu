import Data_FrozenLake2 as dfl2
import numpy as np

def V_star(States, dataParser, gamma):
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
            next_actions_datas = dataParser.get_next_actions(curr_state)
            for next_action_data in next_actions_datas:
                next_action_value, next_action_prob, reward = dataParser.get_action_pi_reward(next_action_data)

                # 获得 动作->状态 转移概率
                next_states_probs = dataParser.get_action_states_probs(next_action_data)
                #next_states_prob = P_as[action_value]
                v_sum = 0
                # 遍历每个转移概率
                for [next_state_value, next_state_prob] in next_states_probs:
                    # math: \sum_{s'} P_{ss'}^a v_{\pi}(s')
                    v_sum += next_state_prob * V_next[next_state_value]
                #end for
                # math: \sum_a \pi(a|s) [R_s^a + \gamma \sum_{s'} P_{ss'}^a v_{\pi}(s')] 
                list_v.append(reward + gamma * v_sum)
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

def Q_star(Actions, dataParser, gamma):
    num_action = len(Actions)
    Q_curr = [0.0] * num_action
    Q_next = [0.0] * num_action
    count = 0
    # 迭代
    while (count < 100):
        # 遍历每个action
        for curr_action in Actions:
            q_curr_sum = 0
            # 获得 动作->状态 转移概率
            reward, next_states_probs = dataParser.get_next_states_probs(curr_action.value)
            if (reward is None):
                continue
            # 遍历每个转移概率求和
            for [next_state_value, next_state_prob] in next_states_probs:
                # 获得 状态->动作 策略概率
                actions_datas = dataParser.get_next_actions(next_state_value)
                list_q = []
                # 求最大值
                for action_data in actions_datas:
                    action, _, _ = dataParser.get_action_pi_reward(action_data)
                    list_q.append(Q_next[action])
                #end for
                # math: \sum_{a'} P_{ss'}^a \max q_{\pi}(s',a') 
                if (len(list_q) > 0):
                    q_curr_sum += next_state_prob * max(list_q) 
            # end for
            # math: R_s^a + \gamma  ( \sum_{a'} P_{ss'}^a \max q_{\pi}(s',a') )
            Q_curr[curr_action.value] = reward + gamma * q_curr_sum
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


def Q_star_from_V_star(Actions, dataParser, gamma, v_star):
    Q_star = {}
    # 遍历每个action
    for curr_action in Actions:
        q_sum = 0
        # 获得 动作->状态 转移概率
        reward, next_states_probs = dataParser.get_next_states_probs(curr_action.value)
        if (reward is None):
            continue
        # 遍历每个转移概率求和
        for [next_state_value, next_state_prob] in next_states_probs:
        # math: \sum_{a'} P_{ss'}^a v_{*}(s') 
            q_sum += next_state_prob * v_star[next_state_value]
        # end for
        # math: R_s^a + \gamma  ( \sum_{a'} P_{ss'}^a \max q_{\pi}(s',a') )
        Q_star[curr_action.name] = reward + gamma * q_sum
    #endfor
    return sorted(Q_star.items())



def find_next_best(Q, start):
    action = None
    value = None
    for q in Q:
        if (q[0].startswith(start)):
            if action is None:
                action = q[0]
                value = q[1]
            else:
                if (q[1] > value):
                    action = q[0]
                    value = q[1]
    return action, value
    

if __name__=="__main__":
    gamma = 0.9
    dataParser = dfl2.DataParser()
    vs = V_star(dfl2.States, dataParser, gamma)
    print(np.round(np.array(vs).reshape(4,4), 2))
    
    Q_star = Q_star_from_V_star(dfl2.Actions, dataParser, gamma, vs)
    for q in Q_star:
        print(q)

    start = "a00"
    count = 0
    while(True):
        action, value = find_next_best(Q_star, start)
        print(action, value)
        if (action is None):
            break
        start = "a" + action.replace(start, "")
        count +=1
        if (count > 8):
            break