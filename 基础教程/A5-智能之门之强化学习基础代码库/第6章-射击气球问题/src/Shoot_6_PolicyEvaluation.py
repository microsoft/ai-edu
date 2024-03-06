import numpy as np
import Shoot_6_DataModel as dataModel
import common.Algo_PolicyEvaluation as algo


if __name__=="__main__":
    max_iteration = 1000# 最大迭代次数
    gamma = 1           # 折扣
    Policy = {          # 策略
        0:[0.4,0.6],    # 在状态 0 时，选择射击红球的概率0.4，选择射击蓝球的概率0.6
        1:[0.4,0.6],    # 在状态 1 时，同上
        2:[0.4,0.6],
        3:[0.4,0.6],
        4:[0.4,0.6],
        5:[0.4,0.6],
        6:[0.4,0.6]     # 可以不定义，因为在终止状态没有动作
    }
    env = dataModel.Env(Policy) # 初始化环境
    V, Q = algo.calculate_VQ_pi(env, gamma, max_iteration)    # 迭代计算V,Q
    V = np.round(V,5)
    Q = np.round(Q,5)
    for i,s in enumerate(env.S):
        print(str.format("状态函数：S{0}({1}):\t{2}", i, s.name, V[s.value]))
        print(str.format("动作函数：{0}:{1}\t{2}:{3}", env.A.Red.name, Q[s.value,env.A.Red.value], env.A.Blue.name, Q[s.value,env.A.Blue.value]))
        print("--------------")
    