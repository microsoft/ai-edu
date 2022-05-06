from enum import Enum

class States(Enum):
     Start = 0       # 开始
     SR0 = 1       # 脱靶
     SR1 = 2       # 小奖
     SR3 = 3       # 大奖
     SB0 = 4       # 脱靶
     SB1 = 5       # 小奖
     T6 = 6       # 终止

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
    # state: {action: [(p, s', r),...]}
    States.Start.value:{                                    # 开始状态（第一枪）
        Actions.Red.value:[                                 # 选择射击红球
            (0.80, States.SR0.value, Rewards.Zero.value ),  # 脱靶的概率:0.80,      转移到状态 SR0(1), 得到0分奖励
            (0.05, States.SR1.value, Rewards.Small.value),  # 误中蓝球的概率:0.05,  转移到状态 SR1(2), 得到1分奖励
            (0.15, States.SR3.value, Rewards.Grand.value),  # 击中红球的概率:0.15,  转移到状态 SR3(3), 得到3分奖励
        ],       
        Actions.Blue.value:[                                # 选择射击蓝球
            (0.40, States.SB0.value, Rewards.Zero.value),   # 脱靶的概率:0.40,      转移到状态 SB0(4), 得到0分奖励
            (0.60, States.SB1.value, Rewards.Small.value)   # 击中蓝球的概率:0.6,   转移到状态 SB1(5), 得到1分奖励
        ]                   
    },
    States.SR0.value:{   # 打红球脱靶
        Actions.Red.value:[                                 # 选择射击红球
            (0.80, States.T6.value, Rewards.Zero.value),    # 第二枪脱靶的概率降低到 0.73
            (0.05, States.T6.value, Rewards.Small.value),   # 误中蓝球的概率不变
            (0.15, States.T6.value, Rewards.Grand.value),   # 第二枪击中红球的概率 0.22,提高了
        ],
        Actions.Blue.value:[                                # 选择射击蓝球
            (0.40, States.T6.value, Rewards.Zero.value),    # 第二枪脱靶的概率不变
            (0.60, States.T6.value, Rewards.Small.value)    # 第二枪击中蓝球概率不变
        ]
    },
    States.SR1.value:{   # 打红球误中小奖
        Actions.Red.value:[                                 # 选择射击红球
            (0.78, States.T6.value, Rewards.Zero.value),    # 
            (0.05, States.T6.value, Rewards.Small.value),
            (0.17, States.T6.value, Rewards.Grand.value),
        ],
        Actions.Blue.value:[                 
            (0.45, States.T6.value, Rewards.Zero.value), 
            (0.55, States.T6.value, Rewards.Small.value)
        ]
    }, 
    States.SR3.value:{   # 打红大奖
        Actions.Red.value:[               
            (0.70, States.T6.value, Rewards.Zero.value), 
            (0.05, States.T6.value, Rewards.Small.value),
            (0.25, States.T6.value, Rewards.Grand.value), 
        ],
        Actions.Blue.value:[                 
            (0.20, States.T6.value, Rewards.Zero.value), 
            (0.80, States.T6.value, Rewards.Small.value)
        ]
    },
    States.SB0.value:{   # 打蓝脱靶
        Actions.Red.value:[                   
            (0.80, States.T6.value, Rewards.Zero.value), 
            (0.05, States.T6.value, Rewards.Small.value),
            (0.15, States.T6.value, Rewards.Grand.value), 
        ],
        Actions.Blue.value:[                 
            (0.40, States.T6.value, Rewards.Zero.value), 
            (0.60, States.T6.value, Rewards.Small.value)
        ]
    },
    States.SB1.value:{   # 打蓝小奖
        Actions.Red.value:[                   
            (0.74, States.T6.value, Rewards.Zero.value), 
            (0.04, States.T6.value, Rewards.Small.value),
            (0.22, States.T6.value, Rewards.Grand.value), 
        ],
        Actions.Blue.value:[               
            (0.25, States.T6.value, Rewards.Zero.value), 
            (0.75, States.T6.value, Rewards.Small.value)
        ]
    },
    States.T6.value:{   # 终止
        Actions.Red.value:[
            (1.0, States.T6.value, 0)
        ],
        Actions.Blue.value:[
            (1.0, States.T6.value, 0)
        ]
    }
}

class Env(object):
    def __init__(self, policy):
        self.nS = len(States)
        self.nA = len(Actions)
        self.S = States
        self.A = Actions
        self.P = P
        self.Policy = policy
        self.end_states = [States.T6.value]

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
    p0 = 0.4 * P[States.Start.value][Actions.Red.value][0][0] \
        + 0.6 * P[States.Start.value][Actions.Blue.value][0][0]
    print("第一枪脱靶", p0)

    # 第一枪脱靶后第二枪也脱靶
    p00 = 0.4 * (0.4 * P[States.SR0.value][Actions.Red.value][0][0] \
               + 0.6 * P[States.SR0.value][Actions.Blue.value][0][0]) \
        + 0.6 * (0.4 * P[States.SB0.value][Actions.Red.value][0][0] \
               + 0.6 * P[States.SB0.value][Actions.Blue.value][0][0])
    # 第一枪脱靶后第二枪中小奖
    p01 = 0.4 * (0.4 * P[States.SR0.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.SR0.value][Actions.Blue.value][1][0]) \
        + 0.6 * (0.4 * P[States.SB0.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.SB0.value][Actions.Blue.value][1][0])
    # 第一枪脱靶后第二枪中大奖
    p02 = 0.4 * (0.4 * P[States.SR0.value][Actions.Red.value][2][0]) \
        + 0.6 * (0.4 * P[States.SB0.value][Actions.Red.value][2][0])
    
    p00 = round(p00,3)
    p01 = round(p01,3)
    p02 = round(p02,3)
    print("\t第二枪脱靶",p00)
    print("\t第二枪小奖",p01)
    print("\t第二枪大奖",p02)
    #assert((p00+p01+p02)==1)

    ########################################

    # 第一枪中小奖的概率
    p1 = 0.4 * P[States.Start.value][Actions.Red.value][1][0] \
        + 0.6 * P[States.Start.value][Actions.Blue.value][1][0]
    print("第一枪小奖", p1)

    # 第一枪小奖后第二枪脱靶
    p10 = 0.4 * (0.4 * P[States.SR1.value][Actions.Red.value][0][0] \
               + 0.6 * P[States.SR1.value][Actions.Blue.value][0][0]) \
        + 0.6 * (0.4 * P[States.SB1.value][Actions.Red.value][0][0] \
               + 0.6 * P[States.SB1.value][Actions.Blue.value][0][0])
    # 第一枪小奖后第二枪中小奖
    p11 = 0.4 * (0.4 * P[States.SR1.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.SR1.value][Actions.Blue.value][1][0]) \
        + 0.6 * (0.4 * P[States.SB1.value][Actions.Red.value][1][0] \
               + 0.6 * P[States.SB1.value][Actions.Blue.value][1][0])
    # 第一枪小奖后第二枪中大奖
    p12 = 0.4 * (0.4 * P[States.SR1.value][Actions.Red.value][2][0]) \
        + 0.6 * (0.4 * P[States.SB1.value][Actions.Red.value][2][0])
    p10 = round(p10,3)
    p11 = round(p11,3)
    p12 = round(p12,3)        
    print("\t第二枪脱靶",p10)
    print("\t第二枪小奖",p11)
    print("\t第二枪大奖",p12)
    #assert((p10+p11+p12)==1)

    ########################################

    # 第一枪中大奖的概率
    p2 = 0.4 * P[States.Start.value][Actions.Red.value][2][0]
    p2 = round(p2,3)
    print("第一枪大奖",p2)

    # 第一枪大奖后第二枪脱靶
    p20 = 0.4 * P[States.SR3.value][Actions.Red.value][0][0] \
        + 0.6 * P[States.SR3.value][Actions.Blue.value][0][0]
    # 第一枪大奖后第二枪中小奖
    p21 = 0.4 * P[States.SR3.value][Actions.Red.value][1][0] \
        + 0.6 * P[States.SR3.value][Actions.Blue.value][1][0]
    # 第一枪大奖后第二枪中大奖
    p22 = 0.4 * P[States.SR3.value][Actions.Red.value][2][0]

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
