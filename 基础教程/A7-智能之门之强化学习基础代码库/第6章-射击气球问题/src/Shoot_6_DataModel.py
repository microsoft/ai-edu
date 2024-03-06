from enum import Enum

# 状态空间
class States(Enum):
     Start = 0      # 开始
     S_Red_R0 = 1   # 选择射击红球Red,但是脱靶R=0
     S_Red_R1 = 2   # 选择射击红球Red,但是误中蓝球R=1小奖
     S_Red_R3 = 3   # 选择射击红球Red,击中R=3大奖
     S_Blue_R0 = 4  # 选择射击蓝球Blue,但是脱靶R=0
     S_Blue_R1 = 5  # 选择射击蓝球Blue,击中R=1小奖
     T = 6          # 终止

# 动作空间
class Actions(Enum):
    Red = 0     # 射击红色小气球
    Blue = 1    # 射击蓝色大气球

# 奖励
class Rewards(Enum):
    Zero = 0
    Small = 1
    Grand = 3

P = {
    # state: {action: [(p, s', r),...]}
    States.Start.value:{                                    # 开始状态（第一枪）
        Actions.Red.value:[                                 # 选择射击红球
            (0.80, States.S_Red_R0.value, Rewards.Zero.value ),  # 脱靶的概率:0.80,      转移到状态 SR0(s1), 得到0分奖励
            (0.05, States.S_Red_R1.value, Rewards.Small.value),  # 误中蓝球的概率:0.05,  转移到状态 SR1(s2), 得到1分奖励
            (0.15, States.S_Red_R3.value, Rewards.Grand.value),  # 击中红球的概率:0.15,  转移到状态 SR3(s3), 得到3分奖励
        ],       
        Actions.Blue.value:[                                # 选择射击蓝球
            (0.40, States.S_Blue_R0.value, Rewards.Zero.value),   # 脱靶的概率:0.40,      转移到状态 SB0(s4), 得到0分奖励
            (0.60, States.S_Blue_R1.value, Rewards.Small.value)   # 击中蓝球的概率:0.6,   转移到状态 SB1(s5), 得到1分奖励
        ]                   
    },
    # 以下为第二枪的选择
    States.S_Red_R0.value:{                                 # 第一枪打红球脱靶
        Actions.Red.value:[                                 # 继续选择射击红球
            (0.80, States.T.value, Rewards.Zero.value),     # 第二枪脱靶概率
            (0.05, States.T.value, Rewards.Small.value),    # 第二枪误中蓝球的概率
            (0.15, States.T.value, Rewards.Grand.value),    # 第二枪击中红球的概率
        ],
        Actions.Blue.value:[                                # 第二枪选择射击蓝球
            (0.40, States.T.value, Rewards.Zero.value),     # 第二枪脱靶的概率不变
            (0.60, States.T.value, Rewards.Small.value)     # 第二枪击中蓝球概率不变
        ]
    },
    States.S_Red_R1.value:{                                 # 第一枪打红球误中蓝球
        Actions.Red.value:[                                 # 继续选择射击红球
            (0.78, States.T.value, Rewards.Zero.value),     # 脱靶率降低
            (0.05, States.T.value, Rewards.Small.value),    # 误中蓝色球
            (0.17, States.T.value, Rewards.Grand.value),    # 击中红球率升高
        ],
        Actions.Blue.value:[                                # 第二枪选择射击蓝球
            (0.45, States.T.value, Rewards.Zero.value),     # 脱靶率升高
            (0.55, States.T.value, Rewards.Small.value)     # 击中蓝球率降低
        ]
    }, 
    States.S_Red_R3.value:{                                 # 第一枪打红球击中大奖
        Actions.Red.value:[                                 # 继续选择射击红球
            (0.70, States.T.value, Rewards.Zero.value),     # 脱靶率降低
            (0.05, States.T.value, Rewards.Small.value),    # 误中蓝色球
            (0.25, States.T.value, Rewards.Grand.value),    # 击中红球率升高
        ],
        Actions.Blue.value:[                                # 第二枪选择射击蓝球                 
            (0.20, States.T.value, Rewards.Zero.value),     # 脱靶率降低
            (0.80, States.T.value, Rewards.Small.value)     # 击中蓝球率提高
        ]
    },
    States.S_Blue_R0.value:{                                # 第一枪打蓝脱靶
        Actions.Red.value:[                                 # 继续选择射击红球
            (0.80, States.T.value, Rewards.Zero.value),     # 脱靶率不变
            (0.05, States.T.value, Rewards.Small.value),    # 误中蓝球
            (0.15, States.T.value, Rewards.Grand.value),    # 击中红球率不变
        ],
        Actions.Blue.value:[                                # 第二枪选择射击蓝球
            (0.40, States.T.value, Rewards.Zero.value),     # 脱靶概率不变
            (0.60, States.T.value, Rewards.Small.value)     # 击中蓝球率不变
        ]
    },
    States.S_Blue_R1.value:{                                # 第一枪打蓝球中小奖
        Actions.Red.value:[                                 # 选择射击红球
            (0.74, States.T.value, Rewards.Zero.value),     # 脱靶率降低
            (0.04, States.T.value, Rewards.Small.value),    # 误中蓝球
            (0.22, States.T.value, Rewards.Grand.value),    # 击中目标红球率升高
        ],
        Actions.Blue.value:[                                # 第二枪选择射击蓝球
            (0.25, States.T.value, Rewards.Zero.value),     # 脱靶率很低
            (0.75, States.T.value, Rewards.Small.value)     # 击中蓝球率很高
        ]
    },
    # States.T.value:{   # 终止
    #     Actions.Red.value:[
    #         (1.0, States.T.value, 0)
    #     ],
    #     Actions.Blue.value:[
    #         (1.0, States.T.value, 0)
    #     ]
    # }
}

class Env(object):
    def __init__(self, policy):
        self.nS = len(States)
        self.nA = len(Actions)
        self.S = States
        self.A = Actions
        self.P = P
        self.Policy = policy
        self.end_states = [States.T.value]

    def get_actions(self, s):
        actions = self.P[s]
        return actions.items()

    def is_end(self,s):
        if s in self.end_states:
            return True
        else:
            return False


if __name__=="__main__":
    Policy = {
        0:[0.4,0.6],    # 在状态 0 时，选择红球的概率0.4，选择蓝球的概率0.6
        1:[0.4,0.6],    # 在状态 1 时，同上
        2:[0.4,0.6],
        3:[0.4,0.6],
        4:[0.4,0.6],
        5:[0.4,0.6]
    }
    env = Env(Policy)    
    # 统计概率
    # 第一枪脱靶的概率
    p0 = Policy[0][0] * P[States.Start.value][Actions.Red.value][0][0] \
       + Policy[0][1] * P[States.Start.value][Actions.Blue.value][0][0]
    print("第一枪脱靶", p0)

    # 第一枪脱靶后第二枪也脱靶
    p00 = Policy[0][0] * (Policy[States.S_Red_R0.value][0] * P[States.S_Red_R0.value][Actions.Red.value][0][0] \
                        + Policy[States.S_Red_R0.value][1] * P[States.S_Red_R0.value][Actions.Blue.value][0][0]) \
        + Policy[0][1] * (0.4 * P[States.S_Blue_R0.value][Actions.Red.value][0][0] \
               + 0.6 * P[States.S_Blue_R0.value][Actions.Blue.value][0][0])
    # 第一枪脱靶后第二枪中小奖
    p01 = Policy[0][0] * (0.4 * P[States.S_Red_R0.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.S_Red_R0.value][Actions.Blue.value][1][0]) \
        + Policy[0][1] * (0.4 * P[States.S_Blue_R0.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.S_Blue_R0.value][Actions.Blue.value][1][0])
    # 第一枪脱靶后第二枪中大奖
    p02 = Policy[0][0] * (0.4 * P[States.S_Red_R0.value][Actions.Red.value][2][0]) \
        + Policy[0][1] * (0.4 * P[States.S_Blue_R0.value][Actions.Red.value][2][0])
    
    p00 = round(p00,3)
    p01 = round(p01,3)
    p02 = round(p02,3)
    print("\t第二枪脱靶",p00)
    print("\t第二枪小奖",p01)
    print("\t第二枪大奖",p02)
    #assert((p00+p01+p02)==1)

    ########################################

    # 第一枪中小奖的概率
    p1 = Policy[0][0] * P[States.Start.value][Actions.Red.value][1][0] \
       + Policy[0][1] * P[States.Start.value][Actions.Blue.value][1][0]
    print("第一枪小奖", p1)

    # 第一枪小奖后第二枪脱靶
    p10 = Policy[0][0] * (0.4 * P[States.S_Red_R1.value][Actions.Red.value][0][0] \
               + 0.6 * P[States.S_Red_R1.value][Actions.Blue.value][0][0]) \
        + Policy[0][1] * (0.4 * P[States.S_Blue_R1.value][Actions.Red.value][0][0] \
               + 0.6 * P[States.S_Blue_R1.value][Actions.Blue.value][0][0])
    # 第一枪小奖后第二枪中小奖
    p11 = Policy[0][0] * (0.4 * P[States.S_Red_R1.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.S_Red_R1.value][Actions.Blue.value][1][0]) \
        + Policy[0][1] * (0.4 * P[States.S_Blue_R1.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.S_Blue_R1.value][Actions.Blue.value][1][0])
    # 第一枪小奖后第二枪中大奖
    p12 = Policy[0][0] * (0.4 * P[States.S_Red_R1.value][Actions.Red.value][2][0]) \
        + Policy[0][1] * (0.4 * P[States.S_Blue_R1.value][Actions.Red.value][2][0])
    p10 = round(p10,3)
    p11 = round(p11,3)
    p12 = round(p12,3)        
    print("\t第二枪脱靶",p10)
    print("\t第二枪小奖",p11)
    print("\t第二枪大奖",p12)
    #assert((p10+p11+p12)==1)

    ########################################

    # 第一枪中大奖的概率
    p2 = Policy[0][0] * P[States.Start.value][Actions.Red.value][2][0]
    p2 = round(p2,3)
    print("第一枪大奖",p2)

    # 第一枪大奖后第二枪脱靶
    p20 = Policy[0][0] * P[States.S_Red_R3.value][Actions.Red.value][0][0] \
        + Policy[0][1] * P[States.S_Red_R3.value][Actions.Blue.value][0][0]
    # 第一枪大奖后第二枪中小奖
    p21 = Policy[0][0] * P[States.S_Red_R3.value][Actions.Red.value][1][0] \
        + Policy[0][1] * P[States.S_Red_R3.value][Actions.Blue.value][1][0]
    # 第一枪大奖后第二枪中大奖
    p22 = Policy[0][0] * P[States.S_Red_R3.value][Actions.Red.value][2][0]

    p20 = round(p20,3)            
    p21 = round(p21,3)            
    p22 = round(p22,3)
    print("\t第二枪脱靶",p20)
    print("\t第二枪小奖",p21)
    print("\t第二枪大奖",p22)
    #assert((p20+p21+p22)==1)
    #assert((p0+p1+p2)==1)

    assert(p00 == p0)
    assert(p01 == p1)
    assert(p02 == p2)
    assert(p10 <= p0)
    assert(p11 >= p1)
    assert(p12 >= p2)
    assert(p20 <= p0)
    assert(p21 >= p1)
    assert(p22 >= p2)
    assert(p21 >= p11)
    assert(p21 >= p21)
