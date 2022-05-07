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
        0:[(0.20, 1, 3), (0.75, 2, 0), (0.05, 3, 1)],       # 第一次击中目标红球的概率=0.2
        1:[(0.40, 4, 0), (0.60, 5, 1)]                      # 第一次击中目标兰球的概率=0.6
    },
    1:{                                                     # 第一次击中目标红球
        0:[(0.25, 6, 3), (0.70, 7, 0), (0.05, 8, 1)],       # 第二次击中红球的概率=0.25，提高了
        1:[(0.35, 9, 0), (0.65, 10, 1)]                     # 第二次击中兰球的概率=0.65，提高了
    },
    2:{                                                     # 第一次打红球脱靶
        0:[(0.20, 11, 3), (0.75, 12, 0), (0.05, 13, 1)],    # 第二次击中红球的概率=0.20，没变化
        1:[(0.40, 14, 0), (0.60, 15, 1)]                    # 第二次击中兰球的概率=0.60，没变化
    }, 
    3:{                                                     # 第一次误中兰球
        0:[(0.18, 16, 3), (0.77, 17, 0), (0.05, 18, 1)],    # 第二次击中红球的概率=0.18，降低了
        1:[(0.45, 19, 0), (0.55, 20, 1)]                    # 第二次击中兰球的概率=0.55，降低了
    },
    4:{                                                     # 第一次打兰球脱靶
        0:[(0.20, 21, 3), (0.75, 22, 0), (0.05, 23, 1)],    # 第二次击中红球的概率=0.20，没变化
        1:[(0.35, 24, 0), (0.65, 25, 1)]                   # 第一次击中兰球的概率=0.65，提高了
    },
    5:{                                                     # 第一次击中目标兰球
        0:[(0.22, 26, 3), (0.73, 27, 0), (0.05, 28, 1)],   # 第二次击中红球的概率=0.22，提高了
        1:[(0.25, 29, 0), (0.75, 30, 1)]                   # 第二次击中兰球的概率=0.75，提高了
    }
}


class Env(object):
    def __init__(self):
        self.nS = 31
        self.nA = 2  # red | blue
        self.P = P
        self.Policy = np.zeros((6, self.nA))
        self.Policy[0:6,0] = np.random.randint(0,2,6)    # 随机生成 0/1 表示动作序号作为初始策略
        self.Policy[0:6,1] = 1 - self.Policy[0:6,0]

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
        else:   # 状态 s6 以下没有后续动作
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


def policy_evaluation(env: Env, gamma):
    V_curr = [0.0] * env.nS
    V_next = [0.0] * env.nS
    Q = copy.deepcopy(env.P)    # 拷贝状态转移结构以存储Q(s,a)值
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for s in range(env.nS):
            v_pi = 0
            # 获得 状态->动作 策略概率
            actions = env.get_actions(s)
            if actions is None:
                continue
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
                v_pi += env.Policy[s,action] * q_pi
                # 顺便把 q 记住，免得以后再算一遍
                Q[s][action] = q_pi
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
    print("迭代次数 = ",count)
    #print(Q)
    return V_next

def policy_improvement(env:Env, V, gamma):
    policy = np.zeros((6, env.nA))    
    for s in range(env.nS):
        actions = env.get_actions(s)
        list_q = [0] * env.nA
        if actions is None:
            continue
        # 遍历每个策略概率
        for action, next_p_s_r in actions:
            q = 0
            for p, s_next, r in next_p_s_r:
                q += p * (r + gamma * V[s_next])
            list_q[action] = q
        best_action = np.argmax(list_q)
        policy[s, best_action] = 1
    return policy    


if __name__=="__main__":
    env = Env()
    print("初始策略:")
    print(env.Policy)
    gamma = 1
    count = 0
    while True:
        count += 1
        print(count)
        print("策略评估")
        V = policy_evaluation(env, gamma)
        print("策略改进")
        new_policy = policy_improvement(env, V, gamma)
        if (new_policy == env.Policy).all():
            break
        else:
            print(new_policy)
            env.Policy = new_policy.copy()
    print("最终策略")    
    print(env.Policy)







