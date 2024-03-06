import common.GridWorld_Model as model

# 状态空间 = 空间宽度 x 空间高度
GridWidth, GridHeight = 5, 5
# 起点，可以多个
StartStates = []
# 终点，可以多个
EndStates = []
# 动作空间
LEFT, DOWN, RIGHT, UP  = 0, 1, 2, 3
Actions = [LEFT, DOWN, RIGHT, UP]
# 初始策略
Policy = []  # empty means random policy 0.25,0.25,0.25,0.25 in GridWorld
# 转移概率: [SlipLeft, MoveFront, SlipRight, SlipBack]
Transition = [0.0, 1.0, 0.0, 0.0]
# 每走一步的奖励值，可以是0或者-1
StepReward = 0
# 特殊奖励 from s->s' then get r, 其中 s,s' 为状态序号，不是坐标位置
SpecialReward = {
    (0,0):-1,       # s0 -> s0 得到-1奖励
    (1,1):-1,
    (2,2):-1,
    (3,3):-1,
    (4,4):-1,
    (5,5):-1,
    (9,9):-1,
    (10,10):-1,
    (14,14):-1,
    (15,15):-1,
    (19,19):-1,
    (20,20):-1,
    (21,21):-1,
    (22,22):-1,
    (23,23):-1,
    (24,24):-1,
    (1,12):+5,
    (21,3):+10
}
# 特殊移动，用于处理类似虫洞场景
SpecialMove = {
    (1,LEFT):   12,
    (1,UP):     12,
    (1,RIGHT):  12,
    (1,DOWN):   12,
    (21,LEFT):   3,
    (21,UP):     3,
    (21,RIGHT):  3,
    (21,DOWN):   3
}
# 墙
Blocks = []

if __name__=="__main__":
    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                    # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    model.print_P(env.P_S_R)
