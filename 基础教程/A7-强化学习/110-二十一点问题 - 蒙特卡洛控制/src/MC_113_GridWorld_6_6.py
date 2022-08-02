import numpy as np
import common.GridWorld_Model as model
import Algorithm.Algo_PolicyValueFunction as algoPi
import common.CommonHelper as helper
import common.DrawQpi as drawQ
import tqdm

# 状态空间 = 空间宽度 x 空间高度
GridWidth, GridHeight = 6, 6
# 起点，可以多个
StartStates = [0]
# 终点，可以多个
EndStates = [35]
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
    (29,35):1,
    (34,35):1
}
# 特殊移动，用于处理类似虫洞场景
SpecialMove = {
}
# 墙
Blocks = []

# MC 策略评估（预测）：每次访问法估算 Q_pi
def MC_EveryVisit_Q_Policy_test(env, episodes, gamma, policy):
    Q_history = []
    nA = env.action_space.n                 # 动作空间
    nS = env.observation_space.n            # 状态空间
    Value = np.zeros((nS, nA))              # G 的总和
    Count = np.ones((nS, nA))              # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s = env.reset()
        Episode = []     # 一幕内的(状态,动作,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, _ = env.step(action)
            Episode.append((s, action, reward))
            s = next_s  # 迭代

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s,a] += G     # 值累加
            Count[s,a] += 1     # 数量加 1

            # greedy policy
            qs = Value[s] / Count[s]
            if np.sum(qs) == 0:
                continue
            policy[s] = 0
            argmax = np.argmax(qs)
            policy[s][argmax] = 1


    Q = Value / Count   # 求均值
 
    return Q




if __name__=="__main__":

    np.random.seed(15)

    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                     # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    #model.print_P(env.P_S_R)
    policy = helper.create_policy(env.nS, env.nA, (0.25, 0.25, 0.25, 0.25))
    gamma = 1
    max_iteration = 1000

    Q = MC_EveryVisit_Q_Policy_test(env, max_iteration, gamma, policy)
    V = helper.calculat_V_from_Q(Q, policy)
    helper.print_seperator_line(helper.SeperatorLines.short, "V 函数")
    print(np.round(V,0).reshape(6,6))
    helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
    print(np.around(Q, 0))
    drawQ.draw(Q, (6,6))
