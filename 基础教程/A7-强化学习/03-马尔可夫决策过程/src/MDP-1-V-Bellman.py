from enum import Enum
import numpy as np

# 状态
class States(Enum):
    Rest = 0
    Game = 1
    Class1 = 2
    Class2 = 3
    Class3 = 4

# 动作
class Actions(Enum):
    Quit = 0
    Play1 = 1
    Play2 = 2
    Study1 = 3
    Study2 = 4
    Pass = 5
    Pub = 6
    Sleep= 7

# 动作奖励
Rewards = [0, -1, -1, -2, -2, 10, 1, 0]

# 状态->动作概率
Pi_sa = np.array([
    # S_Rest -> A_none
    [0, 0, 0, 0, 0, 0, 0, 0],
    # S_Game -> A_Quit, A_Play1
    [0.5, 0.5, 0, 0, 0, 0, 0, 0],
    # S_Class1 -> A_Play2, A_Study1
    [0, 0, 0.5, 0.5, 0, 0, 0, 0],
    # S_Class2 -> A_Study2, A_Sleep
    [0, 0, 0, 0, 0.5, 0, 0, 0.5],
    # S_Class3 -> A_Pass, A_Pub
    [0, 0, 0, 0, 0, 0.5, 0.5, 0]
])

# 动作->状态概率
P_as = np.array([
    # A_Quit -> S_Class1
    [0, 0, 1, 0, 0],
    # A_Play1 -> S_Game
    [0, 1, 0, 0, 0],
    # A_Play2 -> S_Game
    [0, 1, 0, 0, 0],
    # A_Study1 -> S_Class2
    [0, 0, 0, 1, 0],
    # A_Study2 -> S_Class3
    [0, 0, 0, 0, 1],
    # A_Pass -> S_Rest
    [1, 0, 0, 0, 0],
    # A_Pub -> S_Class1, S_Class2, S_Class3
    [0, 0, 0.2, 0.4, 0.4],
    # A_Sleep -> S_None
    [0, 0, 0, 0, 0]
])

def run(gamma):
    num_action = 8
    num_state = 5
    V_curr = [0.0] * 5
    V_next = [0.0] * 5
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for curr_state in States:
            v_curr_sum = 0
            if (curr_state == States.Rest):
                continue
            # 获得 状态->动作 策略概率
            next_actions_prob = Pi_sa[curr_state.value]
            # 遍历每个策略概率
            for action_value, action_prob in enumerate(next_actions_prob):
                if (action_prob == 0.0 and action_value == Actions.Sleep.value):
                    continue
                # 获得 动作->状态 转移概率
                next_states_prob = P_as[action_value]
                v_sum = 0
                # 遍历每个转移概率
                for state_value, state_prob in enumerate(next_states_prob):
                    if (state_prob > 0.0):
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

if __name__=="__main__":
    print(Rewards)
    gamma = 1
    v = run(gamma)
    for start_state in States:
        print(start_state, v[start_state.value])
