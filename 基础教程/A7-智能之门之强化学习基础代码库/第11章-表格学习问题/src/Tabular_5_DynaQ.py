import common.GridWorld_MC_Model as mc_model
from common.Algo_Dyna_Q import DynaQ
import numpy as np
import common.DrawQpi as drawQpi
import common.CommonHelper as helper

# 状态空间 = 空间宽度 x 空间高度
GridWidth, GridHeight = 9, 6
# 起点，可以多个
StartStates = [18]
# 终点，可以多个
EndStates = [8]
# 动作空间
LEFT, DOWN, RIGHT, UP  = 0, 1, 2, 3
Actions = [LEFT, DOWN, RIGHT, UP]
# 转移概率: [SlipLeft, MoveFront, SlipRight, SlipBack]
Transition = [0.0, 1.0, 0.0, 0.0]
# 每走一步的奖励值，可以是0或者-1
StepReward = 0
GoalReward = 1
SpecialReward = {
    20: -1,
    32: -1,
    34: -1,
}
# 特殊移动，用于处理类似虫洞场景
SpecialMove = {}
# 墙
Blocks = [7,11,16,25,29,41,50]

if __name__=="__main__":
    env = mc_model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Transition,                    # 关于动作的参数
        GoalReward, StepReward, SpecialReward,          # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制

    for i in [0, 50]:
        s, _ = env.reset()
        behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        episodes = 4
        ctrl = DynaQ(env, episodes, behavior_policy, alpha=0.1, gamma=0.95, epsilon=0.1, plan_step=i)
        Result = ctrl.run()
        for i, Q in enumerate(Result):
            helper.print_seperator_line(helper.SeperatorLines.middle, "iter: %d" % i)
            drawQpi.drawQ(Q, (6,9), round=8, goal_state=8, end_state=Blocks)
        helper.print_Q(Q, 8, (6,9), helper.SeperatorLines.middle, "Q")

