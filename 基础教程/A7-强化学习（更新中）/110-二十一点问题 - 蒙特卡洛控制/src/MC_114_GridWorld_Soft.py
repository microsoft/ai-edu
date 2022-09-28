import numpy as np
import common.GridWorld_Model as model
import Algorithm.Algo_OptimalValueFunction as algo
import Algorithm.Base_MC_Policy_Iteration as base
import common.CommonHelper as helper
import common.DrawQpi as drawQ
import tqdm

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

class MC_SoftGreedy(base.Policy_Iteration):
    def __init__(self, env, init_policy, gamma, epsilon):
        super().__init__(env, init_policy, gamma)
        self.epsilon = epsilon
        self.best_p = 1 - epsilon + epsilon / self.nA 
        self.other_p = epsilon / self.nA        

    def policy_improvement(self, s):
        # 做策略改进，贪心算法
        if np.min(self.Count[s]) == 0:  # 避免被除数为 0
            return
        self.Q[s] = self.Value[s] / self.Count[s]  # 得到该状态下所有动作的 q 值
        self.policy[s] = self.other_p         # 先设置该状态所有策略为 epsilong/nA
        argmax = np.argmax(self.Q[s])
        self.policy[s, argmax] = self.best_p


if __name__=="__main__":

    np.random.seed(15)

    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                    # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    #model.print_P(env.P_S_R)
    gamma = 0.9
    episodes = 1000

    helper.print_seperator_line(helper.SeperatorLines.long, "试验探索次数")
    epsilons = [0.1, 0.3, 0.5, 0.7]
    
    for epsilon in epsilons:
        helper.print_seperator_line(helper.SeperatorLines.middle, "epsilon="+str(epsilon))
        policy = helper.create_policy(env.nS, env.nA, (0.25, 0.25, 0.25, 0.25))
        mc = MC_SoftGreedy(env, policy, gamma, epsilon)
        Q = mc.policy_iteration(episodes)
        V = helper.calculat_V_from_Q(Q, policy)

        helper.print_seperator_line(helper.SeperatorLines.short, "V 函数")
        print(np.round(V,1).reshape(4,4))
        helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
        print(np.around(Q, 1))
        helper.print_seperator_line(helper.SeperatorLines.short, "策略")
        drawQ.draw(Q, (4,4))

        print(policy)
