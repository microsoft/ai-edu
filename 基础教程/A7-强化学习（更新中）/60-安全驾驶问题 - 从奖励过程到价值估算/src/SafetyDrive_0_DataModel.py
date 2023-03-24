
import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Start = 0           # 出发
    Normal = 1          # 正常行驶
    Pedestrians = 2     # 礼让行人
    DownSpeed = 3       # 闹市减速
    ExceedSpeed = 4     # 超速行驶
    RedLight = 5        # 路口闯灯
    LowSpeed = 6        # 小区慢行
    MobilePhone = 7     # 拨打手机
    Crash = 8           # 发生事故
    Goal = 9            # 安全抵达
    End = 10            # 结束

# 状态转移概率
P = np.array(
    [   #S    N    P    D    E    R    L    M    C    G    E
        [0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # S
        [0.0, 0.0, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1, 0.0, 0.1, 0.0], # N
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # P
        [0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # D
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0], # E
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0], # R
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # L
        [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0], # M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # C
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # G
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # E
    ]
)

# 奖励向量
# |出发|正常行驶|礼让行人|闹市减速|超速行驶|路口闯灯|小区减速|拨打手机|发生事故|安全抵达|结束|
R = [0,  0,      +1,       +1,     -3,      -6,     +1,     -3,      -1,   +5,   0]

class DataModel(object):
    def __init__(self):
        self.P = P                          # 状态转移矩阵
        self.R = R                          # 奖励
        self.S = States                     # 状态集
        self.nS = len(self.S)               # 状态数量
        self.end_states = [self.S.End]      # 终止状态集
    
    # 判断给定状态是否为终止状态
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    # 获得即时奖励，保留此函数可以为将来更复杂的奖励函数做准备
    def get_reward(self, s):
        return self.R[s.value]

    # 根据转移概率前进一步，返回（下一个状态、即时奖励、是否为终止）
    def step(self, s):
        next_s = np.random.choice(self.S, p=self.P[s.value])
        return next_s, self.get_reward(s), self.is_end(next_s)
