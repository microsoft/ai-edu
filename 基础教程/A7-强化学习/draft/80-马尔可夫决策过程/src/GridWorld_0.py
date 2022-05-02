import numpy as np
from enum import Enum
import GridWorldAgent as agent

# 状态空间（尺寸）S，终点目标T，起点S，障碍B，奖励R，动作空间A，转移概率P

# 空间宽度
GridWidth = 4
# 空间高度
GridHeight = 4
# 起点
#Start = [0,0]
# 终点
End = [(0,0)]
# 墙
#Block = [[1,1]]
# end with reward，如果r=0,则StepReward=-1
EndReward = {
    (0,0):-1
}
# 每走一步都-1，如果配置为0，则不减1，而是要在End处得到最终奖励
StepReward = -1

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# 动作空间
class Actions(Enum):
    UP = UP
    RIGHT = RIGHT
    DOWN = DOWN
    LEFT = LEFT


# 转移概率
class Probs(Enum):
    SlipLeft = 0.0
    MoveFront = 1.0
    SlipRight = 0.0
    SlipBack = 0.0


if __name__=="__main__":
    env = agent.GridWorld(GridHeight, GridWidth, Actions, EndReward, Probs, StepReward)
    gamma = 1
    iteration = 1000
    V_pi = agent.V_pi_2array(env, gamma, iteration)
    print(np.reshape(np.round(V_pi,2), (4,4)))

