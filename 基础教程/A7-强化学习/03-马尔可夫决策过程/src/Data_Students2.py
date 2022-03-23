from enum import Enum
import numpy as np

# 状态
class States(Enum):
    Rest = 0
    Game = 1
    Class1 = 2
    Class2 = 3
    Class3 = 4

# 动作
class Actions(Enum):
    Quit = 0
    Play1 = 1
    Play2 = 2
    Study1 = 3
    Study2 = 4
    Pass = 5
    Pub = 6
    Sleep= 7

# 动作奖励
Rewards = [0, -1, -1, -2, -2, 10, 1, 0]

# 状态->动作概率
Pi_sa = np.array([
    # S_Rest -> A_none
    [0, 0, 0, 0, 0, 0, 0, 0],
    # S_Game -> A_Quit, A_Play1
    [0.5, 0.5, 0, 0, 0, 0, 0, 0],
    # S_Class1 -> A_Play2, A_Study1
    [0, 0, 0.5, 0.5, 0, 0, 0, 0],
    # S_Class2 -> A_Study2, A_Sleep
    [0, 0, 0, 0, 0.5, 0, 0, 0.5],
    # S_Class3 -> A_Pass, A_Pub
    [0, 0, 0, 0, 0, 0.5, 0.5, 0]
])

# 动作->状态概率
Pr_as = np.array([
    # A_Quit -> S_Class1
    [0, 0, 1, 0, 0],
    # A_Play1 -> S_Game
    [0, 1, 0, 0, 0],
    # A_Play2 -> S_Game
    [0, 1, 0, 0, 0],
    # A_Study1 -> S_Class2
    [0, 0, 0, 1, 0],
    # A_Study2 -> S_Class3
    [0, 0, 0, 0, 1],
    # A_Pass -> S_Rest
    [1, 0, 0, 0, 0],
    # A_Pub -> S_Class1, S_Class2, S_Class3
    [0, 0, 0.2, 0.4, 0.4],
    # A_Sleep -> S_None
    [0, 0, 0, 0, 0]
])

