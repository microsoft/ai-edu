@@ -1,56 +0,0 @@
from tokenize import Special
import numpy as np
from enum import Enum
import GridWorldAgent2 as agent2

# 状态空间（尺寸）S，终点目标T，起点S，障碍B，奖励R，动作空间A，转移概率P

# 空间宽度
GridWidth = 2
# 空间高度
GridHeight = 2
# 起点
#Start = [0,0]
# 墙
#Block = [[1,1]]
# end with reward，如果r=0,则StepReward=-1
EndStates = [3]

# 每走一步都-1，如果配置为0，则不减1，而是要在End处得到最终奖励
StepReward = 0

SpecialReward = {
    (0,2):-1,
    (1,3):+1,
    (2,3):+1
}

SpecialMove = {}

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# 动作空间
Actions = [UP, RIGHT, DOWN, LEFT]

# 初始策略
Pi = [0.25, 0.25, 0.25, 0.25]

# 转移概率
# SlipLeft, MoveFront, SlipRight, SlipBack
Probs = [0.0, 1.0, 0.0, 0.0]

if __name__=="__main__":
    env = agent2.GridWorld(GridWidth, GridHeight, Actions, SpecialReward, Probs, StepReward, EndStates, SpecialMove, Pi)
    gamma = 1
    iteration = 1000
    V_pi = agent2.V_pi_2array(env, gamma, iteration)
    print(np.reshape(np.round(V_pi,2), (GridWidth, GridHeight)))

    V_star, Q_star = agent2.V_star(env, gamma)
    print("V*")
    print(np.reshape(np.round(V_star,3), (GridWidth,GridHeight)))
    print("Q*")
    agent2.print_P(Q_star)

    policy = agent2.get_policy(env, V_star, gamma)
    agent2.print_policy(policy, (GridWidth, GridHeight))