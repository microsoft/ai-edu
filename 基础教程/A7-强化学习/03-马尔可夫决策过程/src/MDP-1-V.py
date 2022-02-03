from enum import Enum
import numpy as np

# 状态
class States(Enum):
    Rest = 0
    Game = 1
    Class1 = 2
    Class2 = 3
    Class3 = 4

class Actions(Enum):
    Quit = 0
    Play1 = 1
    Play2 = 2
    Study1 = 3
    Study2 = 4
    Pass = 5
    Pub = 6
    Sleep= 7

Rewards = [0, -1, -1, -2, -2, 1, 10, 0]

P_sa = np.array([
    # S_Rest -> A_none
    [0, 0, 0, 0, 0, 0, 0, 0],
    # S_Game -> A_Quit, A_Play1
    [0.5, 0.5, 0, 0, 0, 0, 0, 0],
    # S_Class1 -> A_Play2, A_Study1
    [0, 0, 0.5, 0.5, 0, 0, 0, 0],
    # S_Class2 -> A_Study2, A_Sleep
    [0, 0, 0, 0, 0.5, 0, 0, 0.5],
    # S_Class3 -> A_Pub, A_Pass
    [0, 0, 0, 0, 0, 0.5, 0.5, 0]
])

PI_as = np.array([
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
    # A_Pub -> S_Class1, S_Class2, S_Class3
    [0, 0, 0.2, 0.4, 0.4],
    # A_Pass -> S_Rest
    [1, 0, 0, 0, 0],
    # A_Sleep -> S_None
    [0, 0, 0, 0, 0]
])

def run(gamma):
    num_action = 8
    num_state = 5
    V_curr = [0.0] * 5
    V_next = [0.0] * 5
    count = 0
    while (True):
        for curr_state in States:
            v_pai_sum = 0
            if (curr_state == States.Rest):
                continue
            next_actions_prob = P_sa[curr_state.value]
            for action_value, action_prob in enumerate(next_actions_prob):
                if (action_value == Actions.Sleep.value):
                    continue
                next_states_prob = PI_as[action_value]
                v_sum = 0
                for state_value, state_prob in enumerate(next_states_prob):
                    v_sum += state_prob * V_next[state_value]
                #end for
                v_pai_sum += action_prob * (Rewards[action_value] + gamma * v_sum)
            # end for
            V_curr[curr_state.value] = v_pai_sum
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
    gamma = 0.9
    v = run(gamma)
    for start_state in States:
        print(start_state, v[start_state.value])
