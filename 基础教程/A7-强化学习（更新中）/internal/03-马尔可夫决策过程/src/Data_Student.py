
import numpy as np
from enum import Enum


# 状态
class States(Enum):
    Class1 = 0
    Class2 = 1
    Class3 = 2
    Pass = 3
    Pub = 4
    Play = 5
    Sleep = 6

# 收益向量
# [Class1, Class2, Class3, Pass, Pub, Play, Sleep]
Rewards = [-2, -2, -2, 10, 1, -1, 0]

TransMatrix = np.array(
    [   #Cl1  Cl2  Cl3  Pas  Pub  Ply  Slp
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0], # Class1
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2], # CLass2
        [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0], # Class3
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Pass
        [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0], # Pub
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0], # Play
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Sleep
    ]
)
