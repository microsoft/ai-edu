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
#End = [(0,2),()]
# 墙
#Block = [[1,1]]
# end with reward，如果r=0,则StepReward=-1
EndReward = {
    (2,0):-1,
    (0,2):-1,
    (2,2):-1,
    (3,3):+5,
}
# 每走一步都-1，如果配置为0，则不减1，而是要在End处得到最终奖励
StepReward = 0

LEFT, UP, RIGHT, DOWN,  = 0, 1, 2, 3

# 动作空间
class Actions(Enum):
    LEFT = LEFT
    UP = UP
    RIGHT = RIGHT
    DOWN = DOWN


# 转移概率
class Probs(Enum):
    SlipLeft = 0.1
    MoveFront = 0.7
    SlipRight = 0.2
    SlipBack = 0.0


if __name__=="__main__":
    env = agent.GridWorld(GridHeight, GridWidth, Actions, EndReward, Probs, StepReward)
    gamma = 0.9
    iteration = 1000
    V_pi = agent.V_pi_2array(env, gamma, iteration)
    print(np.reshape(np.round(V_pi,2), (4,4)))

    V_star, Q_star = agent.V_star(env, gamma)
    print(np.reshape(np.round(V_star,2), (4,4)))
    print(Q_star)

    policy = agent.get_policy(env, V_star, gamma)
    print(policy)    
