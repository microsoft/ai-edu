import numpy as np
import GridWorld_Model as model           # 模型逻辑
import Algo_PolicyValueFunction as algo     # 算法实现
import Algo_OptimalValueFunction as algo2

# 状态空间（尺寸）S，终点目标T，起点S，障碍B，奖励R，动作空间A，转移概率P

# 空间宽度
GridWidth = 2
# 空间高度
GridHeight = 2
# 起点
StartStates = []
# 终点
EndStates = []

LEFT, UP, RIGHT, DOWN  = 0, 1, 2, 3
# 动作空间
Actions = [LEFT, UP, RIGHT, DOWN]
Policy = [0.25, 0.25, 0.25, 0.25]
# 转移概率
# SlipLeft, MoveFront, SlipRight, SlipBack
SlipProbs = [0.0, 1.0, 0.0, 0.0]

# 每走一步都-1，如果配置为0，则不减1，而是要在End处得到最终奖励
StepReward = 0
# from s->s', get r
# s,s' 为状态序号，不是坐标位置
SpecialReward = {
    (0,0):-1,
    (1,1):-1,
    (2,2):-1,
    (3,3):-1,
    (0,3):+5
}

# 特殊移动，用于处理类似虫洞场景
SpecialMove = {
    (0,LEFT):   3,
    (0,UP):     3,
    (0,RIGHT):  3,
    (0,DOWN):   3,
}
Blocks = []


if __name__=="__main__":
    env = model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, SlipProbs,                     # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    model.print_P(env.P_S_R)
    gamma = 0.5
    iteration = 1000
    V_pi, Q_pi = algo.calculate_VQ_pi(env, gamma, iteration)
    print("v_pi")
    print(np.reshape(np.round(V_pi,2), (GridWidth,GridHeight)))
    print("q_pi")
    print(np.round(Q_pi,2))

    V_star, Q_star = algo2.calculate_VQ_star(env, gamma, 100)
    
    print("v*=",np.round(V_star,5))
    policy = algo2.get_policy(env, V_star, gamma)
    print("policy")
    print(policy)
    print("q*=")
    print(np.round(Q_star,2))
