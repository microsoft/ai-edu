import numpy as np
import common.GridWorld_Model as model
import Algorithm.Algo_OptimalValueFunction as algo
import Algorithm.Base_MC_Policy_Iteration as base
import common.CommonHelper as helper
import common.DrawQpi as drawQ
import tqdm, math

# 状态空间 = 空间宽度 x 空间高度
GridWidth, GridHeight = 4, 4
# 起点，可以多个
StartStates = list(range(15))
# 终点，可以多个
EndStates = [15]
# 动作空间
LEFT, DOWN, RIGHT, UP  = 0, 1, 2, 3
Actions = [LEFT, DOWN, RIGHT, UP]
# 初始策略
Policy = [0.25, 0.25, 0.25, 0.25]
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
    (7,7):-1,
    (8,8):-1,
    (11,11):-1,
    (12,12):-1,
    (13,13):-1,
    (14,14):-1,
    (11,15):1,
    (14,15):1
}

# 特殊移动，用于处理类似虫洞场景
SpecialMove = {
}

# 墙
Blocks = []

class MC_GLIE_Improved(base.Policy_Iteration):
    def policy_improvement(self, s):
        if np.min(self.Count[s]) == 0:  # 避免被除数为 0
            return        
        epsilon = 1 / (math.log(self.n_episode, 10)+1)
        other_p = epsilon / self.nA
        best_p = 1 - epsilon + epsilon/self.nA
        self.Q[s] = self.Value[s] / self.Count[s]  # 得到该状态下所有动作的 q 值
        self.policy[s] = other_p         # 先设置该状态所有策略为 epsilong/nA
        argmax = np.argmax(self.Q[s])
        self.policy[s, argmax] = best_p
        return


if __name__=="__main__":
    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                    # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    gamma = 0.9
    episodes = 2000

    Qs = []
    for i in range(4):
        policy = helper.create_policy(env.nS, env.nA, (0.25, 0.25, 0.25, 0.25))
        mc = MC_GLIE_Improved(env, policy, gamma)
        Q = mc.policy_iteration(episodes)
        V = helper.calculat_V_from_Q(Q, policy)

        helper.print_seperator_line(helper.SeperatorLines.short, "V 函数")
        print(np.round(V,1).reshape(4,4))
        helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
        print(np.around(Q, 1))
        helper.print_seperator_line(helper.SeperatorLines.short, "策略")
        drawQ.draw(Q, (4,4))

        print(policy)
