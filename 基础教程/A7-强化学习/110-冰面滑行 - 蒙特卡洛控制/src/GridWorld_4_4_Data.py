import numpy as np
import common.GridWorld_Model as model
import Algorithm.Algo_MC_Policy_Evaulation as algo
import common.CommonHelper as helper
import common.DrawQpi as drawQ

# 状态空间 = 空间宽度 x 空间高度
GridWidth, GridHeight = 4, 4
# 起点，可以多个
StartStates = [5,6,9,10]
# 终点，可以多个
EndStates = [0,15]
# 动作空间
LEFT, DOWN, RIGHT, UP  = 0, 1, 2, 3
Actions = [LEFT, DOWN, RIGHT, UP]
# 初始策略
Policy = [0.25, 0.25, 0.25, 0.25]
# 转移概率: [SlipLeft, MoveFront, SlipRight, SlipBack]
Transition = [0.0, 1.0, 0.0, 0.0]
# 每走一步的奖励值，可以是0或者-1
StepReward = -1
# 特殊奖励 from s->s' then get r, 其中 s,s' 为状态序号，不是坐标位置
SpecialReward = {
    (1,0):0,
    (4,0):0,
    (14,15):0,
    (11,15):0
}
# 特殊移动，用于处理类似虫洞场景
SpecialMove = {
}
# 墙
Blocks = []

if __name__=="__main__":

    np.random.seed(15)

    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                     # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    #model.print_P(env.P_S_R)

    policy = helper.create_policy(env.nS, env.nA, (0.25, 0.25, 0.25, 0.25))
    Q = algo.MC_EveryVisit_Q_Policy(env, 20000, 1, policy)
    Q = np.round(Q, 0)
    print(Q)
    drawQ.draw(Q, (4,4))
