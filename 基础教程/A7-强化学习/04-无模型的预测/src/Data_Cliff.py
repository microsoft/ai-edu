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

    Start12 = 12
    Dead13 = 13
    Dead14 = 14
    Dead15 = 15
    Dead16 = 16
    Goal17 = 17
    

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
        Actions.DOWN.value: [(1.0, 12, -100, False)],
        Actions.LEFT.value: [(1.0,  6, -1, False)]
    },
    8:{
        Actions.UP.value:   [(1.0,  2, -1, False)],
        Actions.RIGHT.value:[(1.0,  9, -1, False)],
        Actions.DOWN.value: [(1.0, 12, -100, False)],
        Actions.LEFT.value: [(1.0,  7, -1, False)]
    },
    9:{
        Actions.UP.value:   [(1.0,  3, -1, False)],
        Actions.RIGHT.value:[(1.0, 10, -1, False)],
        Actions.DOWN.value: [(1.0, 12, -100, False)],
        Actions.LEFT.value: [(1.0,  8, -1, False)]
    },
    10:{
        Actions.UP.value:   [(1.0,  4, -1, False)],
        Actions.RIGHT.value:[(1.0, 11, -1, False)],
        Actions.DOWN.value: [(1.0, 12, -100, False)],
        Actions.LEFT.value: [(1.0,  9, -1, False)]
    },
    11:{
        Actions.UP.value:   [(1.0,  5, -1, False)],
        Actions.RIGHT.value:[(1.0, 11, -1, False)],
        Actions.DOWN.value: [(1.0, 17, -1,  True)],
        Actions.LEFT.value: [(1.0, 10, -1, False)]
    },

    12:{    # start
        Actions.UP.value:   [(1.0,  6, -1, False)],
        Actions.RIGHT.value:[(1.0, 12, -100, False)],
        Actions.DOWN.value: [(1.0, 12, -1, False)],
        Actions.LEFT.value: [(1.0, 12, -1, False)]
    },
    13:{
    },
    14:{
    },
    15:{
    },
    16:{
    },
    17:{  # goal
        Actions.UP.value:   [(1.0, 17,  0.0, True)],
        Actions.RIGHT.value:[(1.0, 17,  0.0, True)],
        Actions.DOWN.value: [(1.0, 17,  0.0, True)],
        Actions.LEFT.value: [(1.0, 17,  0.0, True)]
    }
}


class Env(object):
    def __init__(self):
        self.state_space = len(States)
        self.action_space = 4
        self.P = P
        self.States = States
        self.transition = np.array([1.0])

    def reset(self, from_start = True):
        if (from_start):
            return self.States.Start12.value
        else:
            idx = np.random.choice(self.state_space)
            return idx

    def get_actions(self, curr_state: int):
        actions = self.P[curr_state]
        return list(actions.keys())

    def step(self, curr_state: int, action: int):
        probs = self.P[curr_state][action]
        return probs[0]
