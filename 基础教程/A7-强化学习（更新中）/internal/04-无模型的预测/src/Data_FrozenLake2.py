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
    
end_states = [States.Hole2, States.Hole8, States.Hole10, States.Goal15]


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
    # shoule be 0.2
    Left = 0.2
    # shoule be 0.1
    Right = 0.1
    All = 1.0


# Reward
Hole = -1.0
Goal = 5.0
SAFE = 0.0


P={
    States.Start.value:{
        # action :[(状态转移概率, 下一个状态, 奖励值, 是否达到终点),()]
        #Actions.UP.value:   [(Probs.Left.value, 0,  0.0, False), (Probs.Front.value, 0,  0.0, False), (Probs.Right.value, 1,  0.0, False)],
        Actions.RIGHT.value:[(Probs.Left.value, 0,  0.0, False), (Probs.Front.value, 1,  0.0, False), (Probs.Right.value, 4,  0.0, False)],
        Actions.DOWN.value: [(Probs.Left.value, 1,  0.0, False), (Probs.Front.value, 4,  0.0, False), (Probs.Right.value, 0,  0.0, False)],
        #Actions.LEFT.value: [(Probs.Left.value, 4,  0.0, False), (Probs.Front.value, 0,  0.0, False), (Probs.Right.value, 0,  0.0, False)]
    },
    States.Safe1.value:{
        #Actions.UP.value:   [(Probs.Left.value, 0,  0.0, False), (Probs.Front.value, 1,  0.0, False), (Probs.Right.value, 2, Hole,  True)],
        Actions.RIGHT.value:[(Probs.Left.value, 1,  0.0, False), (Probs.Front.value, 2, Hole,  True), (Probs.Right.value, 5,  0.0, False)],
        Actions.DOWN.value: [(Probs.Left.value, 2, Hole,  True), (Probs.Front.value, 5,  0.0, False), (Probs.Right.value, 0,  0.0, False)],
        Actions.LEFT.value: [(Probs.Left.value, 5,  0.0, False), (Probs.Front.value, 0,  0.0, False), (Probs.Right.value, 1,  0.0, False)]
    },
    States.Hole2.value:{
        Actions.UP.value:   [(Probs.All.value, 2,  0.0, True)],
        Actions.RIGHT.value:[(Probs.All.value, 2,  0.0, True)],
        Actions.DOWN.value: [(Probs.All.value, 2,  0.0, True)],
        Actions.LEFT.value: [(Probs.All.value, 2,  0.0, True)]
    },
    States.Safe3.value:{
        #Actions.UP.value:   [(Probs.Left.value, 2, Hole,  True), (Probs.Front.value, 3,  0.0, False), (Probs.Right.value, 3,  0.0, False)],
        #Actions.RIGHT.value:[(Probs.Left.value, 3,  0.0, False), (Probs.Front.value, 3,  0.0, False), (Probs.Right.value, 7,  0.0, False)],
        Actions.DOWN.value: [(Probs.Left.value, 3,  0.0, False), (Probs.Front.value, 7,  0.0, False), (Probs.Right.value, 2, Hole,  True)],
        Actions.LEFT.value: [(Probs.Left.value, 7,  0.0, False), (Probs.Front.value, 2, Hole,  True), (Probs.Right.value, 3,  0.0, False)]
    },
    States.Safe4.value:{
        Actions.UP.value:   [(Probs.Left.value, 4,  0.0, False), (Probs.Front.value, 0,  0.0, False), (Probs.Right.value, 5,  0.0, False)],
        Actions.RIGHT.value:[(Probs.Left.value, 0,  0.0, False), (Probs.Front.value, 5,  0.0, False), (Probs.Right.value, 8, Hole,  True)],
        Actions.DOWN.value: [(Probs.Left.value, 5,  0.0, False), (Probs.Front.value, 8, Hole,  True), (Probs.Right.value, 4,  0.0, False)],
        #Actions.LEFT.value: [(Probs.Left.value, 8, Hole,  True), (Probs.Front.value, 4,  0.0, False), (Probs.Right.value, 0,  0.0, False)]
    },
    States.Safe5.value:{
        Actions.UP.value:   [(Probs.Left.value, 4,  0.0, False), (Probs.Front.value, 1,  0.0, False), (Probs.Right.value, 6,  0.0, False)],
        Actions.RIGHT.value:[(Probs.Left.value, 1,  0.0, False), (Probs.Front.value, 6,  0.0, False), (Probs.Right.value, 9,  0.0, False)],
        Actions.DOWN.value: [(Probs.Left.value, 6,  0.0, False), (Probs.Front.value, 9,  0.0, False), (Probs.Right.value, 4,  0.0, False)],
        Actions.LEFT.value: [(Probs.Left.value, 9,  0.0, False), (Probs.Front.value, 4,  0.0, False), (Probs.Right.value, 1,  0.0, False)]
    },
    States.Safe6.value:{
        Actions.UP.value:   [(Probs.Left.value, 5,  0.0, False), (Probs.Front.value, 2, Hole,  True), (Probs.Right.value, 7,  0.0, False)],
        Actions.RIGHT.value:[(Probs.Left.value, 2, Hole,  True), (Probs.Front.value, 7,  0.0, False), (Probs.Right.value,10, Hole,  True)],
        Actions.DOWN.value: [(Probs.Left.value, 7,  0.0, False), (Probs.Front.value,10, Hole,  True), (Probs.Right.value, 5,  0.0, False)],
        Actions.LEFT.value: [(Probs.Left.value,10, Hole,  True), (Probs.Front.value, 5,  0.0, False), (Probs.Right.value, 2, Hole,  True)]
    },
    States.Safe7.value:{
        Actions.UP.value:   [(Probs.Left.value, 6,  0.0, False), (Probs.Front.value, 3,  0.0, False), (Probs.Right.value, 7,  0.0, False)],
        #Actions.RIGHT.value:[(Probs.Left.value, 3,  0.0, False), (Probs.Front.value, 7,  0.0, False), (Probs.Right.value,11,  0.0, False)],
        Actions.DOWN.value: [(Probs.Left.value, 7,  0.0, False), (Probs.Front.value,11,  0.0, False), (Probs.Right.value, 6,  0.0, False)],
        Actions.LEFT.value: [(Probs.Left.value,11,  0.0, False), (Probs.Front.value, 6,  0.0, False), (Probs.Right.value, 3,  0.0, False)]
    },
    States.Hole8.value:{
        Actions.UP.value:   [(Probs.All.value, 8,  0.0, True)],
        Actions.RIGHT.value:[(Probs.All.value, 8,  0.0, True)],
        Actions.DOWN.value: [(Probs.All.value, 8,  0.0, True)],
        Actions.LEFT.value: [(Probs.All.value, 8,  0.0, True)]
    },
    States.Safe9.value:{
        Actions.UP.value:   [(Probs.Left.value, 8, Hole,  True), (Probs.Front.value, 5,  0.0, False), (Probs.Right.value, 10, Hole,  True)],
        Actions.RIGHT.value:[(Probs.Left.value, 5,  0.0, False), (Probs.Front.value,10, Hole,  True), (Probs.Right.value, 13,  0.0, False)],
        Actions.DOWN.value: [(Probs.Left.value,10, Hole,  True), (Probs.Front.value,13,  0.0, False), (Probs.Right.value, 8,  Hole,  True)],
        Actions.LEFT.value: [(Probs.Left.value,13,  0.0, False), (Probs.Front.value, 8, Hole,  True), (Probs.Right.value, 5,   0.0, False)]
    },
    States.Hole10.value:{
        Actions.UP.value:   [(Probs.All.value, 10,  0.0, True)],
        Actions.RIGHT.value:[(Probs.All.value, 10,  0.0, True)],
        Actions.DOWN.value: [(Probs.All.value, 10,  0.0, True)],
        Actions.LEFT.value: [(Probs.All.value, 10,  0.0, True)]
    },
    States.Safe11.value:{
        Actions.UP.value:   [(Probs.Left.value, 10, Hole,  True), (Probs.Front.value,  7,  0.0, False), (Probs.Right.value, 11,  0.0, False)],
        #Actions.RIGHT.value:[(Probs.Left.value,  7,  0.0, False), (Probs.Front.value, 11,  0.0, False), (Probs.Right.value, 15, Goal,  True)],
        Actions.DOWN.value: [(Probs.Left.value, 11,  0.0, False), (Probs.Front.value, 15, Goal,  True), (Probs.Right.value, 10, Hole,  True)],
        Actions.LEFT.value: [(Probs.Left.value, 15, Goal,  True), (Probs.Front.value, 10, Hole,  True), (Probs.Right.value,  7,  0.0, False)]
    },
    States.Safe12.value:{
        Actions.UP.value:   [(Probs.Left.value, 12,  0.0, False), (Probs.Front.value,  8, Hole,  True), (Probs.Right.value, 13,  0.0, False)],
        Actions.RIGHT.value:[(Probs.Left.value,  8, Hole,  True), (Probs.Front.value, 13,  0.0, False), (Probs.Right.value, 12,  0.0, False)],
        #Actions.DOWN.value: [(Probs.Left.value, 13,  0.0, False), (Probs.Front.value, 12,  0.0, False), (Probs.Right.value, 12,  0.0, False)],
        #Actions.LEFT.value: [(Probs.Left.value, 12,  0.0, False), (Probs.Front.value, 12,  0.0, False), (Probs.Right.value,  8, Hole,  True)]
    },
    States.Safe13.value:{
        Actions.UP.value:   [(Probs.Left.value, 12,  0.0, False), (Probs.Front.value,  9,  0.0, False), (Probs.Right.value, 14,  0.0, False)],
        Actions.RIGHT.value:[(Probs.Left.value,  9,  0.0, False), (Probs.Front.value, 14,  0.0, False), (Probs.Right.value, 13,  0.0, False)],
        #Actions.DOWN.value: [(Probs.Left.value, 14,  0.0, False), (Probs.Front.value, 13,  0.0, False), (Probs.Right.value, 12,  0.0, False)],
        Actions.LEFT.value: [(Probs.Left.value, 13,  0.0, False), (Probs.Front.value, 12,  0.0, False), (Probs.Right.value,  9,  0.0, False)]
    },
    States.Safe14.value:{
        Actions.UP.value:   [(Probs.Left.value, 13,  0.0, False), (Probs.Front.value, 10, Hole,  True), (Probs.Right.value, 15, Goal,  True)],
        Actions.RIGHT.value:[(Probs.Left.value, 10, Hole,  True), (Probs.Front.value, 15, Goal,  True), (Probs.Right.value, 14,  0.0, False)],
        #Actions.DOWN.value: [(Probs.Left.value, 15, Goal,  True), (Probs.Front.value, 14,  0.0, False), (Probs.Right.value, 13,  0.0, False)],
        Actions.LEFT.value: [(Probs.Left.value, 14,  0.0, False), (Probs.Front.value, 13,  0.0, False), (Probs.Right.value, 10, Hole,  True)]
    },
    States.Goal15.value:{
        Actions.UP.value:   [(Probs.All.value, 15,  0.0, True)],
        Actions.RIGHT.value:[(Probs.All.value, 15,  0.0, True)],
        Actions.DOWN.value: [(Probs.All.value, 15,  0.0, True)],
        Actions.LEFT.value: [(Probs.All.value, 15,  0.0, True)]
    }
}


ground_truth = np.array([
    Actions.DOWN.value,  Actions.LEFT.value,                   -1, Actions.DOWN.value,
    Actions.RIGHT.value, Actions.RIGHT.value, Actions.RIGHT.value, Actions.DOWN.value,
                     -1, Actions.DOWN.value,                   -1, Actions.DOWN.value,
    Actions.RIGHT.value, Actions.RIGHT.value, Actions.RIGHT.value,                 -1
])

class Env(object):
    def __init__(self):
        self.state_space = len(States)
        self.action_space = 4
        self.P = P
        self.States = States
        self.EndStates = end_states
        self.transition = np.array([Probs.Left.value, Probs.Front.value, Probs.Right.value])
        self.ground_truth = ground_truth

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
        if (len(probs) == 1):
            return self.P[curr_state][action][0]
        else:
            idx = np.random.choice(3, p=self.transition)
            return self.P[curr_state][action][idx]

    def RMSE(self, value):
        v = value.copy()
        v[v == 0] = -10
        d = np.argmax(v, axis=1)
        d[2] = d[8] = d[10] = d[15] = -1
        return RMSE(d, self.ground_truth)

def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a - b))/b.shape[0])
    return err

if __name__=="__main__":
    env = Env()
    
    done = False
    curr_state_value = env.reset(from_start=True)
    while not done:
        print(curr_state_value)
        actions = env.get_actions(curr_state_value)
        #print(actions)
        idx = np.random.choice(4)
        action=actions[idx]
        print(action)
        p, next_state_value, r, done = env.step(curr_state_value, action)
        print(p,next_state_value,r,done)
        curr_state_value = next_state_value

