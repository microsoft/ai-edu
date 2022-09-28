import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Goal0 = 0
    Safe1 = 1
    Hole2 = 2
    Safe3 = 3
    Safe4 = 4
    Safe5 = 5
    

# 动作 对于4x4的方格，有正反48个动作（再减去进入终点后不能返回的数量）
class Actions(Enum):
    a0001=0x0001
    a0102=0x0102
    a0100=0x0100
    a0201=0x0201

    a0004=0x0004
    a0400=0x0400
    a0105=0x0105
    a0501 = 0x0501 
    a0206=0x0206
    a0602 = 0x0602
    
    a0405 = 0x0405
    a0506 = 0x0506
    a0504 = 0x0504
    a0605 = 0x0605

    
# 向前走动作F时，
# 到达前方s的概率是0.7, 
# 滑到左侧的概率是0.2,
# 滑到左侧的概率是0.1,
# 如果是边角，前方概率不变，越界时呆在原地
Front = 0.7
Left = 0.2
Right = 0.1
# Reward
Hole = -1
Goal = 5

# 数据在数组中的位置语义
Action = 0
ActionPi = 1
Reward = 2
StateProbs = 3

P=[
    [ # state 0: action, pi, reward, [state, prob]
        #[0x0000, 1, Goal, [[0, 1]]],
    ],
    [ # state 1: action, prob, reward, [state, prob]
        [0x0100, 1/3, 0, [[0, Front],[4, Left],[1, Right]]],
        [0x0102, 1/3, Hole, [[2, Front],[1, Left],[5, Right]]],
        [0x0104, 1/3, 0, [[4, Front],[2, Left],[0, Right]]]
    ],
    [ # state 2: action, prob, reward, [state, prob]
        #[0x0201, 1/3, 0, [[1, Front],[6, Left],[2, Right]]],
        #[0x0203, 1/3, 0, [[3, Front],[2, Left],[6, Right]]],
        #[0x0206, 1/3, 0, [[6, Front],[3, Left],[1, Right]]]
        #[0x0202, 1, Hole, [[2, 1]]]
    ],

    #############
    [ # state 3: action, prob, reward, [state, prob]
        [0x0300, 1/2, 0, [[0, Front],[3, Left],[4, Right]]],
        [0x0304, 1/2, 0, [[4, Front],[0, Left],[3, Right]]],
    ],
    [ # state 4: action, prob, reward, [state, prob]
        [0x0401, 1/3, 0, [[1, Front],[3, Left],[5, Right]]],
        [0x0403, 1/3, 0, [[3, Front],[4, Left],[1, Right]]],
        [0x0405, 1/3, 0, [[5, Front],[1, Left],[4, Right]]],
    ],
    [ # state 5: action, prob, reward, [state, prob]
        [0x0502, 1/2, Hole, [[2, Front],[4, Left],[5, Right]]],
        [0x0504, 1/2, 0, [[4, Front],[5, Left],[2, Right]]],
    ],

]

class DataParser(object):
    def get_next_actions(self, curr_state):
        actions_data = P[curr_state.value]
        #print(actions_data)
        return actions_data

    def get_action_pi_reward(self, action_data):
        return action_data[Action], action_data[ActionPi], action_data[Reward]
    
    def get_action_states_probs(self, action_data):
        return action_data[StateProbs]

    def get_next_states_probs(self, action):
        for state in P:
            for actions_data in state:
                if (actions_data[Action] == action):
                    return actions_data[Reward], actions_data[StateProbs]
        return None, None



def V_pi(States, dataParser, gamma):
    num_state = 6
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
    dataParser = DataParser()
    vs = V_pi(States, dataParser, gamma)
    print(np.round(np.array(vs).reshape(2,3), 2))
    Q = Q2_pi(Actions, dataParser, gamma, vs)
    for q in Q:
        print(q, "={:.4f}".format(Q[q]))

