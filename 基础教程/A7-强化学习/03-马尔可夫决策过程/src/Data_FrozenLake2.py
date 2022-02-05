from sunau import AUDIO_FILE_ENCODING_ADPCM_G723_3
import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Start = 0
    Safe1 = 1
    Hole2 = 2
    Safe3 = 3
    Safe4 = 4
    Safe5 = 5
    Safe6 = 6
    Safe7 = 7
    Hole8 = 8
    Safe9 = 9
    Hole10 = 10
    Safe11 = 11
    Safe12 = 12
    Safe13 = 13
    End = 14
    Safe15 = 15
    

# 动作 对于4x4的方格，有正反48个动作（再减去进入终点后不能返回的数量）
class Actions(Enum):
    a0001=0x0001
    a0102=0x0102
    a0203=0x0203
    a0100=0x0100
    a0201=0x0201
    a0302=0x0302

    a0004=0x0004
    a0400=0x0400
    a0105=0x0105
    a0501 = 0x0501 
    a0206=0x0206
    a0602 = 0x0602
    a0307=0x0307
    a0703 = 0x0703
    
    a0405 = 0x0405
    a0506 = 0x0506
    a0607 = 0x0607
    a0504 = 0x0504
    a0605 = 0x0605
    a0706 = 0x0706

    a0408 = 0x0408
    a0804 = 0x0804
    a0509 = 0x0509
    a0905 = 0x0905
    a0610 = 0x0610
    a1006 = 0x1006
    a0711 = 0x0711
    a1107 = 0x1107
    
    a0809 = 0x0809
    a0910 = 0x0910
    a1011 = 0x1011
    a1110 = 0x1110
    a1009 = 0x1009
    a0908 = 0x0908
    
    a0812 = 0x0812
    a1208 = 0x1208
    a0913 = 0x0913
    a1309 = 0x1309
    a1014 = 0x1014
    a1410 = 0x1410
    a1115 = 0x1115
    a1511 = 0x1511

    a1213 = 0x1213
    a1314 = 0x1314
    a1415 = 0x1415

    a1312 = 0x1312 
    a1413 = 0x1413
    a1514 = 0x1514

    
# 向前走动作F时，
# 到达前方s的概率是0.7, 
# 滑到左侧的概率是0.2,
# 滑到左侧的概率是0.1,
# 如果是边角，前方概率不变，越界时呆在原地
F = 0.7
L = 0.2
R = 0.1
H = -1
G = 5

Action = 0
ActionPi = 1
Reward = 2
StateProbs = 3

P=[
    [ # state 0: action, prob, reward, [state, prob]
        [0x0001, 1/2, 0, [[1, F],[0, L],[4, R]]],
        [0x0004, 1/2, 0, [[4, F],[1, L],[0, R]]]
    ],
    [ # state 1: action, prob, reward, [state, prob]
        [0x0100, 1/3, 0, [[0, F],[5, L],[1, R]]],
        [0x0102, 1/3, 0, [[2, F],[1, L],[5, R]]],
        [0x0105, 1/3, H, [[5, F],[2, L],[0, R]]]
    ],
    [ # state 2: action, prob, reward, [state, prob]
        [0x0201, 1/3, 0, [[1, F],[6, L],[2, R]]],
        [0x0203, 1/3, 0, [[3, F],[2, L],[6, R]]],
        [0x0206, 1/3, 0, [[6, F],[3, L],[1, R]]]
    ],
    [ # state 3: action, prob, reward, [state, prob]
        [0x0302, 1/2, 0, [[2, F],[7, L],[3, R]]],
        [0x0307, 1/2, H, [[7, F],[3, L],[2, R]]]
    ],
    #############
    [ # state 4: action, prob, reward, [state, prob]
        [0x0400, 1/3, 0, [[0, F],[4, L],[5, R]]],
        [0x0405, 1/3, H, [[5, F],[0, L],[8, R]]],
        [0x0408, 1/3, 0, [[8, F],[5, L],[4, R]]]
    ],
    [ # state 5: action, prob, reward, [state, prob]
        #[0x0501, 1/4, 0, [[1, F],[4, L],[6, R]]],
        #[0x0506, 1/4, 0, [[6, F],[1, L],[9, R]]],
        #[0x0509, 1/4, 0, [[9, F],[6, L],[4, R]]],
        #[0x0504, 1/4, 0, [[4, F],[9, L],[1, R]]]
    ],
    [ # state 6: action, prob, reward, [state, prob]
        [0x0602, 1/4, 0, [[2, F],[5, L],[7, R]]],
        [0x0607, 1/4, H, [[7, F],[2, L],[10, R]]],
        [0x0610, 1/4, 0, [[10, F],[5, L],[7, R]]],
        [0x0605, 1/4, H, [[5, F],[10, L],[2, R]]]
    ],
    [ # state 7: action, prob, reward, [state, prob]
        #[0x0703, 1/3, 0, [[3, F],[6, L],[7, R]]],
        #[0x0706, 1/3, 0, [[6, F],[11, L],[3, R]]],
        #[0x0711, 1/3, 0, [[11, F],[7, L],[6, R]]]
    ],
    ################
    [ # state 8: action, prob, reward, [state, prob]
        [0x0804, 1/3, 0, [[4, F],[8, L],[9, R]]],
        [0x0809, 1/3, 0, [[9, F],[4, L],[12, R]]],
        [0x0812, 1/3, H, [[12, F],[9, L],[8, R]]]
    ],
    [ # state 9: action, prob, reward, [state, prob]
        [0x0905, 1/4, H, [[5, F],[8, L],[10, R]]],
        [0x0910, 1/4, 0, [[10, F],[5, L],[13, R]]],
        [0x0913, 1/4, 0, [[13, F],[10, L],[8, R]]],
        [0x0908, 1/4, 0, [[8, F],[13, L],[5, R]]]
    ],
    [ # state 10: action, prob, reward, [state, prob]
        [0x1006, 1/4, 0, [[6, F],[9, L],[11, R]]],
        [0x1011, 1/4, 0, [[11, F],[6, L],[14, R]]],
        [0x1014, 1/4, 0, [[14, F],[11, L],[9, R]]],
        [0x1009, 1/4, 0, [[9, F],[14, L],[6, R]]]
    ],
    [ # state 11: action, prob, reward, [state, prob]
        [0x1107, 1/3, H, [[7, F],[10, L],[11, R]]],
        [0x1110, 1/3, 0, [[10, F],[15, L],[7, R]]],
        [0x1115, 1/3, G, [[15, F],[15, L],[10, R]]]
    ],
    ###########
    [ # state 12: action, prob, reward, [state, prob]
        #[0x1208, 1/2, H, [[8, F],[12, L],[13, R]]],
        #[0x1213, 1/2, 0, [[13, F],[8, L],[12, R]]]
    ],
    [ # state 13: action, prob, reward, [state, prob]
        [0x1309, 1/3, 0, [[9, F],[12, L],[14, R]]],
        [0x1312, 1/3, H, [[12, F],[13, L],[9, R]]],
        [0x1314, 1/3, 0, [[14, F],[9, L],[13, R]]]
    ],
    [ # state 14: action, prob, reward, [state, prob]
        [0x1410, 1/3, 0, [[10, F],[13, L],[15, R]]],
        [0x1413, 1/3, 0, [[13, F],[14, L],[10, R]]],
        [0x1415, 1/3, G, [[15, F],[10, L],[14, R]]]
    ],
    [ # state 15: action, prob, reward, [state, prob]
        #[0x1511, 1/2, 0, [[11, F],[14, L],[15, R]]],
        #[0x1514, 1/2, G, [[14, F],[15, L],[11, R]]]
    ]
]

class DataParser(object):
    def get_next_actions(self, curr_state):
        actions_data = P[curr_state.value]
        #print(actions_data)
        return actions_data

    def get_action_pi_reward(self, action_data):
        return action_data[Action], action_data[ActionPi], action_data[Reward]
    
    def get_action_states_probs(self, action_data):
        return action_data[StateProbs]

    def get_next_states_probs(self, action):
        for state in P:
            for actions_data in state:
                if (actions_data[Action] == action):
                    return actions_data[Reward], actions_data[StateProbs]
        return None, None

'''
dataParser = DataParser()
data = dataParser.get_next_actions(0)
print(len(data))
for i in range(len(data)):
    a,p,r = dataParser.get_action_pi_reward(data[i])
    print(a,p,r)
    sp = dataParser.get_action_states_probs(data[i])
    print(sp)
'''