import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Safe0 = 0
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
    Safe11 = 11

    Safe12 = 12
    Safe13 = 13
    Safe14 = 14
    Safe15 = 15
    Safe16 = 16
    Safe17 = 17

    Start = 18
    Dead19 = 19
    Dead20 = 20
    Dead21 = 21
    Dead22 = 22
    Goal = 23

end_states = [States.Dead19, States.Dead20, States.Dead21, States.Dead22, States.Goal]

# 动作 对于方格有4个动作
class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

P={
    0:{
        Actions.UP.value:   [(1.0,  0, -1, False)],
        Actions.RIGHT.value:[(1.0,  1, -1, False)],
        Actions.DOWN.value: [(1.0,  6, -1, False)],
        Actions.LEFT.value: [(1.0,  0, -1, False)]
    },
    1:{
        Actions.UP.value:   [(1.0,  1, -1, False)],
        Actions.RIGHT.value:[(1.0,  2, -1, False)],
        Actions.DOWN.value: [(1.0,  7, -1, False)],
        Actions.LEFT.value: [(1.0,  0, -1, False)]
    },
    2:{
        Actions.UP.value:   [(1.0,  2, -1, False)],
        Actions.RIGHT.value:[(1.0,  3, -1, False)],
        Actions.DOWN.value: [(1.0,  8, -1, False)],
        Actions.LEFT.value: [(1.0,  1, -1, False)]
    },
    3:{
        Actions.UP.value:   [(1.0,  3, -1, False)],
        Actions.RIGHT.value:[(1.0,  4, -1, False)],
        Actions.DOWN.value: [(1.0,  9, -1, False)],
        Actions.LEFT.value: [(1.0,  2, -1, False)]
    },
    4:{
        Actions.UP.value:   [(1.0,  4, -1, False)],
        Actions.RIGHT.value:[(1.0,  5, -1, False)],
        Actions.DOWN.value: [(1.0, 10, -1, False)],
        Actions.LEFT.value: [(1.0,  3, -1, False)]
    },
    5:{
        Actions.UP.value:   [(1.0,  5, -1, False)],
        Actions.RIGHT.value:[(1.0,  5, -1, False)],
        Actions.DOWN.value: [(1.0, 11, -1, False)],
        Actions.LEFT.value: [(1.0,  4, -1, False)]
    },


    6:{
        Actions.UP.value:   [(1.0,  0, -1, False)],
        Actions.RIGHT.value:[(1.0,  7, -1, False)],
        Actions.DOWN.value: [(1.0, 12, -1, False)],
        Actions.LEFT.value: [(1.0,  6, -1, False)]
    },
    7:{
        Actions.UP.value:   [(1.0,  1, -1, False)],
        Actions.RIGHT.value:[(1.0,  8, -1, False)],
        Actions.DOWN.value: [(1.0, 13, -1, False)],
        Actions.LEFT.value: [(1.0,  6, -1, False)]
    },
    8:{
        Actions.UP.value:   [(1.0,  2, -1, False)],
        Actions.RIGHT.value:[(1.0,  9, -1, False)],
        Actions.DOWN.value: [(1.0, 14, -1, False)],
        Actions.LEFT.value: [(1.0,  7, -1, False)]
    },
    9:{
        Actions.UP.value:   [(1.0,  3, -1, False)],
        Actions.RIGHT.value:[(1.0, 10, -1, False)],
        Actions.DOWN.value: [(1.0, 15, -1, False)],
        Actions.LEFT.value: [(1.0,  8, -1, False)]
    },
    10:{
        Actions.UP.value:   [(1.0,  4, -1, False)],
        Actions.RIGHT.value:[(1.0, 11, -1, False)],
        Actions.DOWN.value: [(1.0, 16, -1, False)],
        Actions.LEFT.value: [(1.0,  9, -1, False)]
    },
    11:{
        Actions.UP.value:   [(1.0,  5, -1, False)],
        Actions.RIGHT.value:[(1.0, 11, -1, False)],
        Actions.DOWN.value: [(1.0, 17, -1,  True)],
        Actions.LEFT.value: [(1.0, 10, -1, False)]
    },

    12:{
        Actions.UP.value:   [(1.0,  6, -1, False)],
        Actions.RIGHT.value:[(1.0, 13, -1, False)],
        Actions.DOWN.value: [(1.0, 18, -1, False)],
        Actions.LEFT.value: [(1.0, 12, -1, False)]
    },
    13:{
        Actions.UP.value:   [(1.0,  7, -1, False)],
        Actions.RIGHT.value:[(1.0, 14, -1, False)],
        Actions.DOWN.value: [(1.0, 18, -100, False)],
        Actions.LEFT.value: [(1.0, 12, -1, False)]
    },
    14:{
        Actions.UP.value:   [(1.0,  8, -1, False)],
        Actions.RIGHT.value:[(1.0, 15, -1, False)],
        Actions.DOWN.value: [(1.0, 18, -100, False)],
        Actions.LEFT.value: [(1.0, 13, -1, False)]
    },
    15:{
        Actions.UP.value:   [(1.0,  9, -1, False)],
        Actions.RIGHT.value:[(1.0, 16, -1, False)],
        Actions.DOWN.value: [(1.0, 18, -100, False)],
        Actions.LEFT.value: [(1.0, 14, -1, False)]
    },
    16:{
        Actions.UP.value:   [(1.0, 10, -1, False)],
        Actions.RIGHT.value:[(1.0, 17, -1, False)],
        Actions.DOWN.value: [(1.0, 18, -100, False)],
        Actions.LEFT.value: [(1.0, 15, -1, False)]
    },
    17:{
        Actions.UP.value:   [(1.0, 11, -1, False)],
        Actions.RIGHT.value:[(1.0, 17, -1, False)],
        Actions.DOWN.value: [(1.0, 23, -1,  True)],
        Actions.LEFT.value: [(1.0, 16, -1, False)]
    },


    18:{    # start
        Actions.UP.value:   [(1.0, 12, -1, False)],
        Actions.RIGHT.value:[(1.0, 18, -100, False)],
        Actions.DOWN.value: [(1.0, 18, -1, False)],
        Actions.LEFT.value: [(1.0, 18, -1, False)]
    },
    19:{
    },
    20:{
    },
    21:{
    },
    22:{
    },
    23:{  # goal
        Actions.UP.value:   [(1.0, 23,  0.0, True)],
        Actions.RIGHT.value:[(1.0, 23,  0.0, True)],
        Actions.DOWN.value: [(1.0, 23,  0.0, True)],
        Actions.LEFT.value: [(1.0, 23,  0.0, True)]
    }
}


class Env(object):
    def __init__(self):
        self.state_space = len(States)
        self.action_space = 4
        self.P = P
        self.States = States
        self.EndStates = end_states
        self.transition = np.array([1.0])

    def reset(self, from_start = True):
        if (from_start):
            return self.States.Start.value
        else:
            idx = np.random.choice(self.state_space)
            return idx

    def get_actions(self, curr_state: int):
        actions = self.P[curr_state]
        return list(actions.keys())

    def step(self, curr_state: int, action: int):
        probs = self.P[curr_state][action]
        return probs[0]
