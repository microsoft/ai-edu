import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import Algorithm_MRP as algoMRP
import Random_Walker_MRP as ds

class Action(Enum):
    MOVE_LEFT = 0         # left
    MOVE_RIGHT = 1       # right


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
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # Home
    ]
)


if __name__=="__main__":
    gamma = 1
    vs = algoMRP.Matrix(ds, gamma)
    print(np.round(vs, 3))

    vs = algoMRP.Bellman(ds, gamma)
    print(np.round(vs, 3))

#    vs = algoMRP.MonteCarol(ds, [ds.States.Resturant, ds.States.Home], gamma, 5000)
#    print(np.round(vs, 3))
