import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Start = 0
    Safe1 = 1
    Safe2 = 2
    Safe3 = 3
    Safe4 = 4
    Safe5 = 5
    Safe6 = 6
    Safe7 = 7
    Safe8 = 8
    Safe9 = 9
    Safe10 = 10
    Hole11 = 11
    Hole12 = 12
    Safe13 = 13
    Safe14 = 14
    End = 15

# 动作 对于4x4的方格，有正反48个动作（再减去进入终点后不能返回的数量）
class Actions(Enum):
    a01=0
    a04=1
    a10=2
    a12=3
    a15=4
    a21=5
    a23=6
    a26=7
    a32=8
    a37=9
    a40=10
    
# 向前走动作F时，
# 到达前方s的概率是0.7, 
# 滑到左侧的概率是0.2,
# 滑到左侧的概率是0.1,
# 如果是边角，前方概率不变，越界时呆在原地
F = 0.7
L = 0.2
R = 0.1

Action = 0
Prob = 1
Reward = 2

P=[
    [ # state 0: action, prob, reward, [state, prob]
        [0x0001, 1/2, 0, [[1, F],[0, L],[4, R]]],
        [0x0004, 1/2, 0, [[4, F],[1, L],[0, R]]]
    ],
    [ # state 1: action, prob, reward, [state, prob]
        [0x0100, 1/3, 0, [[0, F],[5, L],[1, R]]],
        [0x0102, 1/3, 0, [[2, F],[1, L],[5, R]]],
        [0x0105, 1/3, 0, [[5, F],[2, L],[0, R]]]
    ],
    [ # state 2: action, prob, reward, [state, prob]
        [0x0201, 1/3, 0, [[1, F],[6, L],[2, R]]],
        [0x0203, 1/3, 0, [[3, F],[2, L],[6, R]]],
        [0x0206, 1/3, 0, [[6, F],[3, L],[1, R]]]
    ],
    [ # state 3: action, prob, reward, [state, prob]
        [0x0302, 1/2, 0, [[2, F],[7, L],[3, R]]],
        [0x0307, 1/2, 0, [[7, F],[3, L],[2, R]]]
    ],


    [ # state 4: action, prob, reward, [state, prob]
        [0x0400, 1/3, 0, [[0, F],[4, L],[5, R]]],
        [0x0405, 1/3, 0, [[5, F],[0, L],[8, R]]],
        [0x0408, 1/3, 0, [[8, F],[5, L],[4, R]]]
    ],
    [ # state 4: action, prob, reward, [state, prob]
        [0x0501, 1/4, 0, [[1, F],[4, L],[6, R]]],
        [0x0506, 1/4, 0, [[6, F],[1, L],[9, R]]],
        [0x0509, 1/4, 0, [[9, F],[6, L],[4, R]]],
        [0x0504, 1/4, 0, [[4, F],[9, L],[1, R]]]
    ],
]

class DataParser(object):
    def __init__(self):
      pass

    def get_next_actions(self, curr_state):
        actions_data = P[curr_state]
        print(actions_data)

data = DataParser()
data.get_next_actions(0)
print(len(data))
print(data[Action], data[Prob])
