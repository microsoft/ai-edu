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
        self.nS = 6
        self.nA = 2
        self.A = Actions
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
