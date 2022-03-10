from sre_parse import State
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
    

# 动作 对于方格有4个动作
class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# 向前走动作F时，
# 到达前方s的概率是0.7, 
# 滑到左侧的概率是0.2,
# 滑到左侧的概率是0.1,
# 如果是边角，前方概率不变，越界时呆在原地    
class Probs(Enum):
    Front = 0.7
    Left = 0.2
    Right = 0.1
    All = 1.0


# Reward
Hole = -1
Goal = 5
SAFE = 0



# 数据在数组中的位置语义
Action = 0
ActionPi = 1
Reward = 2
StateProbs = 3

P={
    States.Start:{
        Actions.UP:   [(Probs.Left, 0,  0.0, False), (Probs.Front, 0,  0.0, False), (Probs.Right, 1,  0.0, False)],
        Actions.RIGHT:[(Probs.Left, 0,  0.0, False), (Probs.Front, 1,  0.0,  True), (Probs.Right, 4,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 1,  0.0, False), (Probs.Front, 4,  0.0, False), (Probs.Right, 0,  0.0, False)],
        Actions.LEFT: [(Probs.Left, 4,  0.0, False), (Probs.Front, 0,  0.0, False), (Probs.Right, 0,  0.0, False)]
    },
    States.Safe1:{
        Actions.UP:   [(Probs.Left, 0,  0.0, False), (Probs.Front, 1,  0.0, False), (Probs.Right, 2, Hole,  True)],
        Actions.RIGHT:[(Probs.Left, 1,  0.0, False), (Probs.Front, 2, Hole,  True), (Probs.Right, 5,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 2, Hole,  True), (Probs.Front, 5,  0.0, False), (Probs.Right, 0,  0.0, False)],
        Actions.LEFT: [(Probs.Left, 5,  0.0, False), (Probs.Front, 0,  0.0, False), (Probs.Right, 1,  0.0, False)]
    },
    States.Hole2:{
        Actions.UP:   [(Probs.All, 2,  0.0, True)],
        Actions.RIGHT:[(Probs.All, 2,  0.0, True)],
        Actions.DOWN: [(Probs.All, 2,  0.0, True)],
        Actions.LEFT: [(Probs.All, 2,  0.0, True)]
    },
    States.Safe3:{
        Actions.UP:   [(Probs.Left, 2, Hole,  True), (Probs.Front, 3,  0.0, False), (Probs.Right, 3,  0.0, False)],
        Actions.RIGHT:[(Probs.Left, 3,  0.0, False), (Probs.Front, 3,  0.0, False), (Probs.Right, 7,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 3,  0.0, False), (Probs.Front, 7,  0.0, False), (Probs.Right, 2, Hole,  True)],
        Actions.LEFT: [(Probs.Left, 7,  0.0, False), (Probs.Front, 2, Hole,  True), (Probs.Right, 3,  0.0, False)]
    },
    States.Safe4:{
        Actions.UP:   [(Probs.Left, 4,  0.0, False), (Probs.Front, 0,  0.0, False), (Probs.Right, 5,  0.0, False)],
        Actions.RIGHT:[(Probs.Left, 0,  0.0, False), (Probs.Front, 5,  0.0, False), (Probs.Right, 8, Hole,  True)],
        Actions.DOWN: [(Probs.Left, 5,  0.0, False), (Probs.Front, 8, Hole,  True), (Probs.Right, 4,  0.0, False)],
        Actions.LEFT: [(Probs.Left, 8, Hole,  True), (Probs.Front, 4,  0.0, False), (Probs.Right, 0,  0.0, False)]
    },
    States.Safe5:{
        Actions.UP:   [(Probs.Left, 4,  0.0, False), (Probs.Front, 1,  0.0, False), (Probs.Right, 6,  0.0, False)],
        Actions.RIGHT:[(Probs.Left, 1,  0.0, False), (Probs.Front, 6,  0.0, False), (Probs.Right, 9,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 6,  0.0, False), (Probs.Front, 9,  0.0, False), (Probs.Right, 4,  0.0, False)],
        Actions.LEFT: [(Probs.Left, 9,  0.0, False), (Probs.Front, 4,  0.0, False), (Probs.Right, 1,  0.0, False)]
    },
    States.Safe6:{
        Actions.UP:   [(Probs.Left, 5,  0.0, False), (Probs.Front, 2, Hole,  True), (Probs.Right, 7,  0.0, False)],
        Actions.RIGHT:[(Probs.Left, 2, Hole,  True), (Probs.Front, 7,  0.0, False), (Probs.Right,10, Hole,  True)],
        Actions.DOWN: [(Probs.Left, 7,  0.0, False), (Probs.Front,10, Hole,  True), (Probs.Right, 5,  0.0, False)],
        Actions.LEFT: [(Probs.Left,10, Hole,  True), (Probs.Front, 5,  0.0, False), (Probs.Right, 2, Hole,  True)]
    },
    States.Safe7:{
        Actions.UP:   [(Probs.Left, 6,  0.0, False), (Probs.Front, 3,  0.0, False), (Probs.Right, 7,  0.0, False)],
        Actions.RIGHT:[(Probs.Left, 3,  0.0, False), (Probs.Front, 7,  0.0, False), (Probs.Right,11,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 7,  0.0, False), (Probs.Front,11,  0.0, False), (Probs.Right, 6,  0.0, False)],
        Actions.LEFT: [(Probs.Left,11,  0.0, False), (Probs.Front, 6,  0.0, False), (Probs.Right, 3,  0.0, False)]
    },
    States.Hole8:{
        Actions.UP:   [(Probs.All, 8,  0.0, True)],
        Actions.RIGHT:[(Probs.All, 8,  0.0, True)],
        Actions.DOWN: [(Probs.All, 8,  0.0, True)],
        Actions.LEFT: [(Probs.All, 8,  0.0, True)]
    },
    States.Safe9:{
        Actions.UP:   [(Probs.Left, 8, Hole,  True), (Probs.Front, 5,  0.0, False), (Probs.Right, 10, Hole,  True)],
        Actions.RIGHT:[(Probs.Left, 5,  0.0, False), (Probs.Front,10, Hole,  True), (Probs.Right, 13,  0.0, False)],
        Actions.DOWN: [(Probs.Left,10, Hole,  True), (Probs.Front,13,  0.0, False), (Probs.Right, 8,  Hole,  True)],
        Actions.LEFT: [(Probs.Left,13,  0.0, False), (Probs.Front, 8, Hole,  True), (Probs.Right, 5,  0.0, False)]
    },
    States.Hole10:{
        Actions.UP:   [(Probs.All, 10,  0.0, True)],
        Actions.RIGHT:[(Probs.All, 10,  0.0, True)],
        Actions.DOWN: [(Probs.All, 10,  0.0, True)],
        Actions.LEFT: [(Probs.All, 10,  0.0, True)]
    },
    States.Safe11:{
        Actions.UP:   [(Probs.Left, 10, Hole,  True), (Probs.Front,  7,  0.0, False), (Probs.Right, 11,  0.0, False)],
        Actions.RIGHT:[(Probs.Left,  7,  0.0, False), (Probs.Front, 11,  0.0, False), (Probs.Right, 15, Goal, False)],
        Actions.DOWN: [(Probs.Left, 11,  0.0, False), (Probs.Front, 15, Goal, False), (Probs.Right, 10, Hole,  True)],
        Actions.LEFT: [(Probs.Left, 15, Goal, False), (Probs.Front, 10, Hole,  True), (Probs.Right,  7,  0.0, False)]
    },
    States.Safe12:{
        Actions.UP:   [(Probs.Left, 12,  0.0, False), (Probs.Front,  8, Hole,  True), (Probs.Right, 13,  0.0, False)],
        Actions.RIGHT:[(Probs.Left,  8, Hole,  True), (Probs.Front, 13,  0.0, False), (Probs.Right, 12,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 13,  0.0, False), (Probs.Front, 12,  0.0, False), (Probs.Right, 12,  0.0, False)],
        Actions.LEFT: [(Probs.Left, 12,  0.0, False), (Probs.Front, 12,  0.0, False), (Probs.Right,  8, Hole,  True)]
    },
    States.Safe13:{
        Actions.UP:   [(Probs.Left, 12,  0.0, False), (Probs.Front,  9,  0.0, False), (Probs.Right, 14,  0.0, False)],
        Actions.RIGHT:[(Probs.Left,  9,  0.0, False), (Probs.Front, 14,  0.0, False), (Probs.Right, 13,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 14,  0.0, False), (Probs.Front, 13,  0.0, False), (Probs.Right, 12,  0.0, False)],
        Actions.LEFT: [(Probs.Left, 13,  0.0, False), (Probs.Front, 12,  0.0, False), (Probs.Right,  9,  0.0, False)]
    },
    States.Safe14:{
        Actions.UP:   [(Probs.Left, 13,  0.0, False), (Probs.Front, 10, Hole,  True), (Probs.Right, 15, Goal, False)],
        Actions.RIGHT:[(Probs.Left, 10, Hole,  True), (Probs.Front, 15, Goal, False), (Probs.Right, 14,  0.0, False)],
        Actions.DOWN: [(Probs.Left, 15, Goal,  True), (Probs.Front, 14,  0.0, False), (Probs.Right, 13,  0.0, False)],
        Actions.LEFT: [(Probs.Left, 14,  0.0, False), (Probs.Front, 13,  0.0, False), (Probs.Right, 10, Hole,  True)]
    },
    States.Goal15:{
        Actions.UP:   [(Probs.All, 15,  0.0, True)],
        Actions.RIGHT:[(Probs.All, 15,  0.0, True)],
        Actions.DOWN: [(Probs.All, 15,  0.0, True)],
        Actions.LEFT: [(Probs.All, 15,  0.0, True)]
    }
}



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
        [0x1115, 1/3, Goal, [[15, Front],[15, Left],[10, Right]]]
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

class Data_FrozenLake2(object):
    def __init__(self):
        self.num_states = len(States)
        self.num_actions = len(Actions)

    def step(self, curr_state):
        actions_data = P[curr_state.value]
        p = []
        for action_data in actions_data:
            p.append(action_data[ActionPi])
        next_action_idx = np.random.choice(len(actions_data), p=p)
        action = actions_data[next_action_idx][Action]
        reward = actions_data[next_action_idx][Reward]
        p = []
        tmp = actions_data[next_action_idx][StateProbs]
        for t in tmp:
            p.append(t[1])
        next_state_idx = np.random.choice(len(tmp), p=p)
        next_state_value = tmp[next_state_idx][0]
        return Actions(action), States(next_state_value), reward

    def is_end_state(self, curr_state: States):
        if (curr_state in [States.Hole2, States.Hole10, States.Hole8, States.Goal15]):
            return True
        else:
            return False



