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
    Safe14 = 14
    Goal15 = 15
    

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
Front = 0.7
Left = 0.2
Right = 0.1
# Reward
Hole = -1
Goal = 5

# 数据在数组中的位置语义
Action = 0
ActionPi = 1
Reward = 2
StateProbs = 3

P=[
    [ # state 0: action, pi, reward, [state, prob]
        [0x0001, 1/2, 0, [[1, Front],[0, Left],[4, Right]]],
        [0x0004, 1/2, 0, [[4, Front],[1, Left],[0, Right]]]
    ],
    [ # state 1: action, prob, reward, [state, prob]
        [0x0100, 1/3, 0, [[0, Front],[5, Left],[1, Right]]],
        [0x0102, 1/3, Hole, [[2, Front],[1, Left],[5, Right]]],
        [0x0105, 1/3, 0, [[5, Front],[2, Left],[0, Right]]]
    ],
    [ # state 2: action, prob, reward, [state, prob]
        #[0x0201, 1/3, 0, [[1, Front],[6, Left],[2, Right]]],
        #[0x0203, 1/3, 0, [[3, Front],[2, Left],[6, Right]]],
        #[0x0206, 1/3, 0, [[6, Front],[3, Left],[1, Right]]]
        [0x0202, 1, Hole, [[2, 1]]]
    ],
    [ # state 3: action, prob, reward, [state, prob]
        [0x0302, 1/2, Hole, [[2, Front],[7, Left],[3, Right]]],
        [0x0307, 1/2, 0, [[7, Front],[3, Left],[2, Right]]]
    ],
    #############
    [ # state 4: action, prob, reward, [state, prob]
        [0x0400, 1/3, 0, [[0, Front],[4, Left],[5, Right]]],
        [0x0405, 1/3, 0, [[5, Front],[0, Left],[8, Right]]],
        [0x0408, 1/3, Hole, [[8, Front],[5, Left],[4, Right]]]
    ],
    [ # state 5: action, prob, reward, [state, prob]
        [0x0501, 1/4, 0, [[1, Front],[4, Left],[6, Right]]],
        [0x0504, 1/4, 0, [[4, Front],[9, Left],[1, Right]]],
        [0x0506, 1/4, 0, [[6, Front],[1, Left],[9, Right]]],
        [0x0509, 1/4, 0, [[9, Front],[6, Left],[4, Right]]]
    ],
    [ # state 6: action, prob, reward, [state, prob]
        [0x0602, 1/4, Hole, [[2, Front],[5, Left],[7, Right]]],
        [0x0605, 1/4, 0, [[5, Front],[10, Left],[2, Right]]],
        [0x0607, 1/4, 0, [[7, Front],[2, Left],[10, Right]]],
        [0x0610, 1/4, Hole, [[10, Front],[5, Left],[7, Right]]],
    ],
    [ # state 7: action, prob, reward, [state, prob]
        [0x0703, 1/3, 0, [[3, Front],[6, Left],[7, Right]]],
        [0x0706, 1/3, 0, [[6, Front],[11, Left],[3, Right]]],
        [0x0711, 1/3, 0, [[11, Front],[7, Left],[6, Right]]]
    ],
    ################
    [ # state 8: action, prob, reward, [state, prob]
        #[0x0804, 1/3, 0, [[4, Front],[8, Left],[9, Right]]],
        #[0x0809, 1/3, 0, [[9, Front],[4, Left],[12, Right]]],
        #[0x0812, 1/3, 0, [[12, Front],[9, Left],[8, Right]]]
        [0x0808, 1, Hole, [[8, 1]]]
    ],
    [ # state 9: action, prob, reward, [state, prob]
        [0x0905, 1/4, 0, [[5, Front],[8, Left],[10, Right]]],
        [0x0908, 1/4, Hole, [[8, Front],[13, Left],[5, Right]]],
        [0x0910, 1/4, Hole, [[10, Front],[5, Left],[13, Right]]],
        [0x0913, 1/4, 0, [[13, Front],[10, Left],[8, Right]]]
    ],
    [ # state 10: action, prob, reward, [state, prob]
        #[0x1006, 1/4, 0, [[6, Front],[9, Left],[11, Right]]],
        #[0x1011, 1/4, 0, [[11, Front],[6, Left],[14, Right]]],
        #[0x1014, 1/4, 0, [[14, Front],[11, Left],[9, Right]]],
        #[0x1009, 1/4, 0, [[9, Front],[14, Left],[6, Right]]]
        [0x1010, 1, Hole, [[10, 1]]]
    ],
    [ # state 11: action, prob, reward, [state, prob]
        [0x1107, 1/3, 0, [[7, Front],[10, Left],[11, Right]]],
        [0x1110, 1/3, Hole, [[10, Front],[15, Left],[7, Right]]],
        [0x1115, 1/3, 0, [[15, Front],[15, Left],[10, Right]]]
    ],
    ###########
    [ # state 12: action, prob, reward, [state, prob]
        [0x1208, 1/2, Hole, [[8, Front],[12, Left],[13, Right]]],
        [0x1213, 1/2, 0, [[13, Front],[8, Left],[12, Right]]]
    ],
    [ # state 13: action, prob, reward, [state, prob]
        [0x1309, 1/3, 0, [[9, Front],[12, Left],[14, Right]]],
        [0x1312, 1/3, 0, [[12, Front],[13, Left],[9, Right]]],
        [0x1314, 1/3, 0, [[14, Front],[9, Left],[13, Right]]]
    ],
    [ # state 14: action, prob, reward, [state, prob]
        [0x1410, 1/3, Hole, [[10, Front],[13, Left],[15, Right]]],
        [0x1413, 1/3, 0, [[13, Front],[14, Left],[10, Right]]],
        [0x1415, 1/3, Goal, [[15, Front],[10, Left],[14, Right]]]
        #[0x1414, 1, Goal, [[14, 1]]]
    ],
    [ # state 15: action, prob, reward, [state, prob]
        #[0x1511, 1/2, 0, [[15, Front],[14, Left], [15, Right]]],
        #[0x1514, 1/2, 0, [[14, Front],[15, Left],[11, Right]]]
        [0x1515, 1, Goal, [[15, 1]]]
    ]

]

class DataParser(object):
    def get_next_actions(self, curr_state):
        actions_data = P[curr_state.value]
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


