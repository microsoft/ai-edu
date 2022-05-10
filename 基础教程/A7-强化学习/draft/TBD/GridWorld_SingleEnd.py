@@ -1,39 +0,0 @@
import numpy as np
from enum import Enum
import GridWorldAgent2 as agent2

# 状态空间（尺寸）S，终点目标T，起点S，障碍B，奖励R，动作空间A，转移概率P

# 空间宽度
GridWidth = 4
# 空间高度
GridHeight = 4
# 起点
#Start = [0,0]
# 终点
EndStates = [15]
# 墙
#Block = [[1,1]]
# 每走一步都-1，如果配置为0，则不减1，而是要在End处得到最终奖励
StepReward = -1

SpecialReward = {}

SpecialMove = {}

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

Actions = [UP, RIGHT, DOWN, LEFT]

# 转移概率
# SlipLeft, MoveFront, SlipRight, SlipBack
Probs = [0.0, 1.0, 0.0, 0.0]


if __name__=="__main__":
    env = agent2.GridWorld(GridWidth, GridHeight, Actions, SpecialReward, Probs, StepReward, EndStates, SpecialMove)
    gamma = 1
    iteration = 3
    V_pi = agent2.V_pi_2array(env, gamma, iteration)
    print(np.reshape(np.round(V_pi,2), (4,4)))
