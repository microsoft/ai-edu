@@ -1,67 +0,0 @@
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
#End = [(0,2),()]
# 墙
#Block = [[1,1]]
# end with reward，如果r=0,则StepReward=-1

SpecialReward = {
    (1,2):-1,
    (3,2):-1,
    (6,2):-1,
    (4,8):-1,
    (9,8):-1,
    (11,8):-1,
    (6,10):-1,
    (9,10):-1,
    (11,10):-1,
    (14,10):-1,
    (14,15):+5,
    (11,15):+5,
}

EndStates = [2,8,10,15]

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# 每走一步都-1，如果配置为0，则不减1，而是要在End处得到最终奖励
StepReward = 0

SpecialMove = {}

# 动作空间
Actions = [UP, RIGHT, DOWN, LEFT]

# 转移概率
# SlipLeft, MoveFront, SlipRight, SlipBack
Probs = [0.2, 0.7, 0.1, 0.0]


if __name__=="__main__":
    env = agent2.GridWorld(GridWidth, GridHeight, Actions, SpecialReward, Probs, StepReward, EndStates, SpecialMove)
    #agent2.print_P(env.P)
    gamma = 0.9
    iteration = 1000
    print("V_pi")
    V_pi = agent2.V_pi_2array(env, gamma, iteration)
    print(np.reshape(np.round(V_pi,3), (GridWidth,GridHeight)))

    V_star, Q_star = agent2.V_star(env, gamma)
    print("V*")
    print(np.reshape(np.round(V_star,3), (GridWidth,GridHeight)))
    print("Q*")
    agent2.print_P(Q_star)

    policy = agent2.get_policy(env, V_star, gamma)
    agent2.print_policy(policy, (GridWidth, GridHeight))