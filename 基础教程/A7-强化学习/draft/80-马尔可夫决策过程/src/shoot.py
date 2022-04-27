from sre_parse import State
import numpy as np
from enum import Enum

# 状态空间
class States(Enum):
    Start = 0       # 开始
    Grand = 1       # 大奖
    Miss  = 2       # 脱靶
    Small = 3       # 小奖
#    End = 4

# 动作空间
class Actions(Enum):
    Red = 0     # 红色小气球，可以中大奖
    Blue = 1    # 蓝色大气球，可以中小奖

# 奖励
class Rewards(Enum):
    Zero = 0
    Small = 1
    Grand = 3
    

P = {
    States.Start:{
        Actions.Red:[
            (0.25, States.Grand, Rewards.Grand.value),
            (0.7,  States.Miss,  Rewards.Zero.value),
            (0.05, States.Small, Rewards.Small.value)],
        Actions.Blue:[
            (0.2, States.Miss,  Rewards.Zero.value),
            (0.8, States.Small, Rewards.Small.value)]
    },
    States.Grand:{
        Actions.Red:[
            (0.25, States.Grand, Rewards.Grand.value),
            (0.7,  States.Miss,  Rewards.Zero.value),
            (0.05, States.Small, Rewards.Small.value)],
        Actions.Blue:[
            (0.2, States.Miss,  Rewards.Zero.value),
            (0.8, States.Small, Rewards.Small.value)]
    },
    States.Miss:{
        Actions.Red:[
            (0.25, States.Grand, Rewards.Grand.value),
            (0.7,  States.Miss,  Rewards.Zero.value),
            (0.05, States.Small, Rewards.Small.value)],
        Actions.Blue:[
            (0.2, States.Miss,  Rewards.Zero.value),
            (0.8, States.Small, Rewards.Small.value)]
    },
    States.Small:{
        Actions.Red:[
            (0.25, States.Grand, Rewards.Grand.value),
            (0.7,  States.Miss,  Rewards.Zero.value),
            (0.05, States.Small, Rewards.Small.value)],
        Actions.Blue:[
            (0.2, States.Miss,  Rewards.Zero.value),
            (0.8, States.Small, Rewards.Small.value)]
    }
}


class Env(object):
    def __init__(self):
        self.N = len(States)
        self.action_space = 4
        self.P = P
        self.S = States
        self.Policy = {Actions.Red:0.4, Actions.Blue:0.6}
        #self.end_states = [self.S.End]
        #self.trans = np.array([Probs.Left.value, Probs.Front.value, Probs.Right.value])

    def reset(self, from_start = True):
        if (from_start):
            return self.States.Start.value
        else:
            idx = np.random.choice(self.state_space)
            return idx

    def get_actions(self, curr_state):
        actions = self.P[curr_state]
        return actions.items()

    def step(self, curr_state, action):
        probs = self.P[curr_state][action]
        if (len(probs) == 1):
            return self.P[curr_state][action][0]
        else:
            idx = np.random.choice(3, p=self.transition)
            return self.P[curr_state][action][idx]



def V_pi(env: Env, gamma):
    V_curr = [0.0] * env.N
    V_next = [0.0] * env.N
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for curr_state in env.S:
            v_sum = 0
            # 获得 状态->动作 策略概率
            actions = env.get_actions(curr_state)
            # 遍历每个策略概率
            for action, next_p_s_r in actions:
                # 获得 动作->状态 转移概率
                q_sum = 0
                # 遍历每个转移概率
                for p, s, r in next_p_s_r:
                    # math: \sum_{s'} P_{ss'}^a v_{\pi}(s')
                    q_sum += p * r + gamma * p * V_next[s.value]
                #end for
                # math: \sum_a \pi(a|s) [R_s^a + \gamma \sum_{s'} P_{ss'}^a v_{\pi}(s')] 
                
                v_sum += env.Policy[action] * q_sum
            # end for
            
            V_curr[curr_state.value] = v_sum
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
    env = Env()
    v = V_pi(env, 0.9)
    print(v)
    exit(0)
    actions = env.get_actions(States.Start)
    print(actions)
    for action,trans in actions.items():
        print(action)
        print(trans)