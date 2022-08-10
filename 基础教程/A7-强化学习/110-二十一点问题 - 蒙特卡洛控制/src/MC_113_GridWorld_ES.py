import numpy as np
import common.GridWorld_Model as model
import Algorithm.Algo_OptimalValueFunction as algo
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
Transition = [0.1, 0.8, 0.1, 0.0]
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

# MC 策略评估（预测）：每次访问法估算 Q_pi
def MC_EveryVisit_Q_Policy_test(env, episodes, gamma, policy, exploration):
    nA = env.action_space.n                 # 动作空间
    nS = env.observation_space.n            # 状态空间
    Value = np.zeros((nS, nA))              # G 的总和
    Count = np.zeros((nS, nA))              # G 的数量
    Q = np.zeros((nS, nA))
    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s = env.reset()
        Episode = []     # 一幕内的(状态,动作,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, _ = env.step(action)
            Episode.append((s, action, reward))
            # if (s == next_s and episode >=exploration):
            #     print(s, action, policy[s])
            #     print(Q[s])
            #     print(Value[s])
            #     print(Count[s])
            #     print(policy[s])               
            s = next_s  # 迭代

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s,a] += G     # 值累加
            Count[s,a] += 1     # 数量加 1
            # 判断该状态下被选择动作的最小次数
            if np.min(Count[s]) <= exploration:
                continue        # 如果次数不够，则不做策略改进
            # 做策略改进，贪心算法
            Q[s] = Value[s] / Count[s]  # 得到该状态下所有动作的 q 值
            policy[s] = 0               # 先设置该状态所有策略为 0（后面再把有效的动作设置为非 0）
            argmax = np.argwhere(Q[s]==np.max(Q[s]))    # 取最大值所在的位置（可能不止一个）
            p = 1 / argmax.shape[0]                     # 概率平均分配给几个最大值
            for arg in argmax:                          # 每个最大值位置都设置为相同的概率
                policy[s][arg[0]] = p

    Count[Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
    Q = Value / Count   # 求均值
    return Q


if __name__=="__main__":

    np.random.seed(5)

    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                    # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    #model.print_P(env.P_S_R)
    gamma = 0.9
    max_iteration = 500

    helper.print_seperator_line(helper.SeperatorLines.long, "精确解")

    V_real,Q_real = algo.calculate_VQ_star(env, gamma, max_iteration)
    helper.print_seperator_line(helper.SeperatorLines.short, "V 函数值")
    print(np.round(V_real,1).reshape(4,4))
    helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数值")
    print(np.around(Q_real, 1))
    helper.print_seperator_line(helper.SeperatorLines.short, "策略")
    drawQ.draw(Q_real, (4,4))

    ####################################

    helper.print_seperator_line(helper.SeperatorLines.long, "试验探索次数")
    explorations = [20,40,60,80]
    for exploration in explorations:
        helper.print_seperator_line(helper.SeperatorLines.middle, "探索次数="+str(exploration))
        policy = helper.create_policy(env.nS, env.nA, (0.25, 0.25, 0.25, 0.25))
        Q = MC_EveryVisit_Q_Policy_test(env, max_iteration, gamma, policy, exploration)
        V = helper.calculat_V_from_Q(Q, policy)
        #helper.print_seperator_line(helper.SeperatorLines.short, "V 函数")
        #print(np.round(V,1).reshape(4,4))
        #helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
        #print(np.around(Q, 1))
        #helper.print_seperator_line(helper.SeperatorLines.short, "策略")
        #drawQ.draw(Q, (4,4))
        #helper.print_seperator_line(helper.SeperatorLines.short, "策略概率值")
        #print(policy)

        Qs = []
        for i in range(10):
            q = MC_EveryVisit_Q_Policy_test(env, max_iteration, gamma, policy, max_iteration)
            Qs.append(q)
        Q = np.array(Qs).mean(axis=0) 
        V = helper.calculat_V_from_Q(Q, policy) 
        helper.print_seperator_line(helper.SeperatorLines.short, "V 函数")
        print(np.round(V,1).reshape(4,4))
        helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
        print(np.around(Q, 1))
        helper.print_seperator_line(helper.SeperatorLines.short, "策略")
        drawQ.draw(Q, (4,4))

        #helper.print_seperator_line(helper.SeperatorLines.short, "RMSE误差")
        #print("状态价值函数误差=", helper.RMSE(V_real, V))
        #print("动作价值函数误差=", helper.RMSE(Q_real, Q))

    ####################################################
    """
    helper.print_seperator_line(helper.SeperatorLines.long, "试验采样次数与误差的关系")
    exploration = 100
    iterations = [200,500,1000,2000]
    for max_iteration in iterations:
        policy = helper.create_policy(env.nS, env.nA, (0.25, 0.25, 0.25, 0.25))
        helper.print_seperator_line(helper.SeperatorLines.middle, "采样次数="+str(max_iteration))
        Q = MC_EveryVisit_Q_Policy_test(env, max_iteration, gamma, policy, exploration)
        V = helper.calculat_V_from_Q(Q, policy)

        Qs = []
        for i in range(10):
            q = MC_EveryVisit_Q_Policy_test(env, max_iteration, gamma, policy, max_iteration)
            Qs.append(q)
        Q = np.array(Qs).mean(axis=0) 
        V = helper.calculat_V_from_Q(Q, policy)
        helper.print_seperator_line(helper.SeperatorLines.short, "V 函数")
        print(np.round(V,1).reshape(4,4))
        helper.print_seperator_line(helper.SeperatorLines.short, "Q 函数")
        print(np.around(Q, 1))
        helper.print_seperator_line(helper.SeperatorLines.short, "策略")
        drawQ.draw(Q, (4,4))

        helper.print_seperator_line(helper.SeperatorLines.short, "RMSE误差")
        print("状态价值函数误差=", helper.RMSE(V_real, V))
        print("动作价值函数误差=", helper.RMSE(Q_real, Q))
    """
