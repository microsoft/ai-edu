import numpy as np
import common.GridWorld_Model as model
import common.CommonHelper as helper
import common.DrawQpi as drawQ
import Algorithm.Base_MC_Policy_Iteration as algoMC
import Algorithm.Algo_MC_Policy_Evaulation as algoV

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

class MC_E_Greedy(algoMC.Policy_Iteration):
    def __init__(self, env, policy, gamma:float, rough:int, final:int, epsilon:float):
        super().__init__(env, policy, gamma, rough, final)
        self.epsilon = epsilon
        self.other_p = self.epsilon / self.nA
        self.best_p = 1 - self.epsilon + self.epsilon / self.nA
    
    def policy_improvement(self, Q):
        print(np.sum(Q))

        for s in range(self.nS):
            if s in env.EndStates:
                self.policy[s] = 0
            else:
                max_A = np.max(Q[s])
                argmax_A = np.where(Q[s] == max_A)[0]
                A = np.random.choice(argmax_A)
                self.policy[s] = self.other_p
                self.policy[s,A] = self.best_p

        return self.policy

if __name__=="__main__":
    gamma = 1
    rough_episodes = 500
    final_episodes = 10000
    epsilon = 0.1

    np.random.seed(15)

    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                     # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制

    policy = helper.create_policy(env.nS, env.nA, (0.25,0.25,0.25,0.25))
    policy_0 = policy.copy()
    env.reset()
    algo = MC_E_Greedy(env, policy, gamma, rough_episodes, final_episodes, epsilon)
    Q, policy = algo.policy_iteration()
    env.close()
    
    print("------ 最优动作价值函数 -----")
    Q=np.round(Q,0)
    print(Q)
    drawQ.draw(Q,(4,4))
    print(policy)

    env.reset()
    V = algoV.MC_EveryVisit_V_Policy(env, 10000, gamma, policy_0)
    print(np.reshape(V, (4,4)))
    env.reset()
    V = algoV.MC_EveryVisit_V_Policy(env, 10000, gamma, policy)
    print(np.reshape(V, (4,4)))

    V = helper.calculat_V_from_Q(Q, policy)
    print(np.reshape(V, (4,4)))
