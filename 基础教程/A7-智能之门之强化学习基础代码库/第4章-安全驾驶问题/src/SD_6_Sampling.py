import numpy as np
import common.SimpleModel as Model
import common.Algo_Sampling as Samplier
from enum import Enum
import time

# 状态
class States(Enum):
    Start = 0           # 出发
    Normal = 1          # 正常行驶
    Pedestrians = 2     # 礼让行人
    Downtown = 3        # 闹市减速
    ExceedSpeed = 4     # 超速行驶
    RedLight = 5        # 路口闯灯
    LowSpeed = 6        # 小区慢行
    MobilePhone = 7     # 拨打手机
    Crash = 8           # 发生事故
    Goal = 9            # 安全抵达
    End = 10            # 结束

# 状态转移概率
P = np.array(
    [   #S    N    P    D    E    R    L    M    C    G    E
        [0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # S
        [0.0, 0.0, 0.2, 0.1, 0.1, 0.1, 0.3, 0.1, 0.0, 0.1, 0.0], # N
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # P
        [0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # D
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0], # E
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0], # R
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # L
        [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0], # M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # C
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # G
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # E
    ]
)

# 奖励向量,期望奖励方式
#   |出发|正常行驶|礼让行人|闹市减速|超速行驶|路口闯灯|小区减速|拨打手机|发生事故|安全抵达|结束|
R = [0,   0,      +1,     +1,     -3,      -6,     +1,     -3,      -1,     +5,     0]


if __name__=="__main__":
    start = time.time()
    episodes = 10000        # 计算 10000 次的试验的均值作为数学期望值
    gammas = [0, 0.5, 0.9, 1]    # 折扣因子
    model = Model.SimpleModel(States, P, R2 = R)
    for gamma in gammas:
        print("gamma=", gamma)
        V = {}
        for s in model.S:   # 遍历每个状态
            print("s=", s.name)
            v = Samplier.Sampling(model, s, episodes, gamma) # 采样计算价值函数
            V[s] = v            # 保存到字典中
        # 打印输出
        print("gamma =", gamma)
        for key, value in V.items():
            print(str.format("{0}:\t{1}", key.name, value))
    end = time.time()
    print("耗时 :", end-start)
