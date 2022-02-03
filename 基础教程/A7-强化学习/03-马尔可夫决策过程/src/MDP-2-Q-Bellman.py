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
    Q_curr = [0.0] * num_action
    Q_next = [0.0] * num_action
    count = 0
    # 迭代
    while (True):
        # 遍历每个action
        for curr_action in Actions:
            q_curr_sum = 0
            if (curr_action == Actions.Sleep):
                continue
            # 获得 动作->状态 转移概率
            next_states_prob = P_as[curr_action.value]
            # 遍历每个转移概率求和
            for state_value, state_prob in enumerate(next_states_prob):
                if (state_prob == 0.0 or state_value == States.Rest.value):
                    continue
                # 获得 状态->动作 策略概率
                next_actions_prob = Pi_sa[state_value]
                q_sum = 0
                # 遍历每个动作概率求和
                for action_value, action_prob in enumerate(next_actions_prob):
                    if (action_prob > 0.0):
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

if __name__=="__main__":
    print(Rewards)
    gamma = 1
    v = run(gamma)
    for action in Actions:
        print(action, v[action.value])
