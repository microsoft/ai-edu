import numpy as np
from enum import Enum
import copy

# 状态空间
Num_States = 31

#class States(Enum):
#     Start = 0       # 开始
#     Grand = 1       # 大奖
#     Miss  = 2       # 脱靶
#     Small = 3       # 小奖


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
    # curr state
    0:{                                                     # 开始
        # action:[(p, s', r), (p, s', r)]
        1:[(0.20, 1, 3), (0.75, 2, 0), (0.05, 3, 1)],       # 第一次击中目标红球的概率=0.2
        2:[(0.40, 4, 0), (0.60, 5, 1)]                      # 第一次击中目标兰球的概率=0.6
    },
    1:{                                                     # 第一次击中目标红球
        3:[(0.25, 6, 3), (0.70, 7, 0), (0.05, 8, 1)],       # 第二次击中红球的概率=0.25，提高了
        4:[(0.35, 9, 0), (0.65, 10, 1)]                     # 第二次击中兰球的概率=0.65，提高了
    },
    2:{                                                     # 第一次打红球脱靶
        5:[(0.20, 11, 3), (0.75, 12, 0), (0.05, 13, 1)],    # 第二次击中红球的概率=0.20，没变化
        6:[(0.40, 14, 0), (0.60, 15, 1)]                    # 第二次击中兰球的概率=0.60，没变化
    }, 
    3:{                                                     # 第一次误中兰球
        7:[(0.18, 16, 3), (0.77, 17, 0), (0.05, 18, 1)],    # 第二次击中红球的概率=0.18，降低了
        8:[(0.45, 19, 0), (0.55, 20, 1)]                    # 第二次击中兰球的概率=0.55，降低了
    },
    4:{                                                     # 第一次打兰球脱靶
        9:[(0.20, 21, 3), (0.75, 22, 0), (0.05, 23, 1)],    # 第二次击中红球的概率=0.20，没变化
        10:[(0.35, 24, 0), (0.65, 25, 1)]                   # 第一次击中兰球的概率=0.65，提高了
    },
    5:{                                                     # 第一次击中目标兰球
        11:[(0.22, 26, 3), (0.73, 27, 0), (0.05, 28, 1)],   # 第二次击中红球的概率=0.22，提高了
        12:[(0.25, 29, 0), (0.75, 30, 1)]                   # 第二次击中兰球的概率=0.75，提高了
    }
}


class Env(object):
    def __init__(self):
        self.N = 31
        self.action_space = 2
        self.P = P
        self.Policy = {0:0.4, 1:0.6}
        #self.end_states = [self.S.End]
        #self.trans = np.array([Probs.Left.value, Probs.Front.value, Probs.Right.value])

    def reset(self, from_start = True):
        if (from_start):
            return self.States.Start.value
        else:
            idx = np.random.choice(self.state_space)
            return idx

    def get_actions(self, s):
        if (s < 6):
            actions = self.P[s]
            return actions.items()
        else:
            return None

    def get_states(self, a):
        for s, actions in self.P.items():
            if actions.__contains__(a):
                return actions[a]

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
    Q = copy.deepcopy(env.P)    # 拷贝状态转移结构以存储Q(s,a)值
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for s in range(env.N):
            v_pi = 0
            # 获得 状态->动作 策略概率
            actions = env.get_actions(s)
            if actions is not None:
                # 遍历每个策略概率
                for action, next_p_s_r in actions:
                    # 获得 动作->状态 转移概率
                    q_pi = 0
                    # 遍历每个转移概率,以计算 q_pi
                    for p, s_next, r in next_p_s_r:
                        # 式2.1 math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
                        q_pi += p * (r + gamma * V_curr[s_next])
                    #end for
                    # 式5 math: \sum_a \pi(a|s) q_\pi (s,a)
                    Q[s][action] = q_pi
                    v_pi += env.Policy[1-action%2] * q_pi
                # end for
            V_next[s] = v_pi
        #endfor
        # 检查收敛性
        if np.allclose(V_next, V_curr):
            break
        # 把 V_curr 赋值给 V_next
        V_curr = V_next.copy()
        count += 1
    # end while
    print(count)
    print(Q)
    return V_next


def Q_pi_from_V_pi(V, env:Env, gamma):
    num_action = 12 # 1~12
    Q = [0.0] * (num_action + 1)    # 0 is not used
    # 遍历每个action
    for action in range(1,13):
        q_sum = 0
        # 获得 动作->状态 转移概率
        states = env.get_states(action)
        # 遍历每个转移概率求和
        for p, s_next, r in states:
            # math: \sum_{s'} P_{ss'}^a v_{\pi}(s') 
            q_sum += p * r + gamma * p * V[s_next]
        # end for
        # math: q_{\pi}(s,a)=R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s')
        Q[action] = q_sum
    #endfor
    return Q


if __name__=="__main__":
    env = Env()
    V_pi = V_pi(env, 1)
    print(np.round(V_pi,4))

    Q_pi = Q_pi_from_V_pi(V_pi, env, 1)
    print(np.round(Q_pi,4))


    exit(0)
    actions = env.get_actions(States.Start)
    print(actions)
    for action,trans in actions.items():
        print(action)
        print(trans)