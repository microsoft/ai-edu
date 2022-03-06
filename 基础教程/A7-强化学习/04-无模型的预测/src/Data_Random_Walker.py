from random import choice
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import Algorithm_MRP as algoMRP
import Data_Random_Walker as ds

'''
   [ ]-0-(A)-0-(B)-0-(C)-0-(D)-0-(E)-1-[ ]
s = 0     1     2     3     4     5     6
'''

# 状态
class States(Enum):
    Resturant = 0
    RoadA = 1
    RoadB = 2
    RoadC = 3
    RoadD = 4
    RoadE = 5
    Home = 6

Rewards = [0,0,0,0,0,0,1]

TransMatrix = np.array(
    [#to: R    A    B    C    D    E    H
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # Resturant
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],    # from C
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]     # Home
    ]
)


class Data_Random_Walker(object):
    def __init__(self):
        self.num_states = len(States)

    def get_states(self) -> States:
        return States
        
    def get_states_count(self) -> int:
        return len(States)

    def step(self, curr_state: States):
        next_state_value = np.random.choice(self.num_states, p=TransMatrix[curr_state.value])
        reward = Rewards[next_state_value]
        return States(next_state_value), reward

    def get_reward(self, curr_state: States):
        return Rewards[curr_state.value]

    def is_end_state(self, curr_state: States):
        if (curr_state in [States.Home, States.Resturant]):
            return True
        else:
            return False

if __name__=="__main__":
    gamma = 1
    vs = algoMRP.Matrix(ds, gamma)
    print(np.round(vs*6, 3))

    vs = algoMRP.Bellman(ds, gamma)
    print(np.round(vs*6, 3))

    #vs = algoMRP.MonteCarol(ds, [ds.States.Resturant, ds.States.Home], gamma, 5000)
    #print(np.round(vs, 3))
