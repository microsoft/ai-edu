
import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Game = 0
    Class1 = 1
    Class2 = 2
    Class3 = 3
    Pass = 4
    Rest = 5
    End = 6

# 奖励向量
# [Game, Class1, Class2, Class3, Pass, Rest, End]
Rewards = [-1, -2, -2, -2, 10, 1, 0]

# 状态转移概率
P = np.array(
    [   #Game Cl1  Cl2  Cl3  Pass Rest End
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
    ]
)

ground_truth = [-22.54, -12.54, 1.46, 4.32, 10.00, 0.80, 0.00]

class Data(object):
    def __init__(self):
        self.P = P
        self.R = Rewards
        self.S = States
        self.num_states = len(self.S)
        self.end_states = [self.S.End]
        self.Y = ground_truth
    
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    def get_reward(self, s):
        return self.R[s.value]

    def step(self, curr_s):
        next_s_value = np.random.choice(self.num_states, p=self.P[curr_s.value])
        next_s = States(next_s_value)
        return next_s, self.get_reward(next_s), self.is_end(next_s)
