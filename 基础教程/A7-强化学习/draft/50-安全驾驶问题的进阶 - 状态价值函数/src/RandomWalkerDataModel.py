import numpy as np
from enum import Enum
import tqdm
import math

class States(Enum):
    Start = 0
    A = 1
    B = 2
    C = 3
    D = 4
    Home = 5
    End = 6

P = np.array(
    [  # S    A    B    C    D    H    end   
        [0.5, 0.5, 0,   0,   0  , 0  , 0  ], # S
        [0.5, 0,   0.5, 0,   0,   0,   0  ], # A
        [0,   0.5, 0,   0.5, 0,   0,   0  ], # B
        [0,   0,   0.5, 0,   0.5, 0,   0  ], # C
        [0,   0,   0,   0.5, 0,   0.5, 0  ], # D
        [0,   0,   0,   0,   0,   0,   1.0], # Home
        [0,   0,   0,   0,   0,   0,   1.0], # end
])

# 状态奖励值
#     S,   A,  B,  C,  D, H, End
R1 = [ 0,  0,  0,  0,  0, 1, 0]
R2 = [-1, -1, -1, -1, -1, 0, 0]

# 过程奖励值
R3 = {
    States.Start: {
        States.Start:   -1,
        States.A:       -1
    },
    States.A: {
        States.Start:   -1,
        States.B:       -1
    },
    States.B: {
        States.A:       -1,
        States.C:       -1
    },
    States.C: {
        States.B:       -1,
        States.D:       -1
    },
    States.D: {
        States.C:       -1,
        States.Home:    0
    },
    States.Home: {
        States.End:     0
    },
}

R3 = [
    (-1)*0.5+(-1)*0.5,  # S
    (-1)*0.5+(-1)*0.5,  # A
    (-1)*0.5+(-1)*0.5,  # B
    (-1)*0.5+(-1)*0.5,  # C
    (-1)*0.5+(0)*0.5,   # D
    0,                  # Home
    0,                  # End
]
R4 = [
    (-1)*0.5+(-1)*0.5,  # S
    (-1)*0.5+(-1)*0.5,  # A
    (-1)*0.5+(-1)*0.5,  # B
    (-1)*0.5+(-1)*0.5,  # C
    (-1)*0.5+(-1)*0.5,  # D
    1*1.0,              # Home
    0,                  # End
]

class DataModel(object):
    def __init__(self, R):
        self.P = P                          # 状态转移矩阵
        self.R = R                          # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)                # 状态数量
        self.end_states = [self.S.End]      # 终止状态集

    # 判断给定状态是否为终止状态
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    def get_reward(self, s):
        return self.R[s.value]

    def get_reward(self, s, s_next):
        return self.R[s.value]

    # 根据转移概率前进一步，返回（下一个状态、即时奖励、是否为终止）
    def step(self, s):
        next_s = np.random.choice(self.S, p=self.P[s.value])
        return next_s, self.get_reward(next_s), self.is_end(next_s)


def Matrix(dataModel, gamma):
    I = np.eye(dataModel.N) * (1+1e-7)
    #I = np.eye(dataModel.N)
    tmp1 = I - gamma * dataModel.P
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, dataModel.R)
    return vs

def print_V(V):
    print(V)
    vv = np.around(V,3)
    for s in dataModel.S:
        print(str.format("\t\t{0}:\t{1}", s.name, vv[s.value]))

# 多次采样获得回报 G 的数学期望，即状态价值函数 V
def Sampling(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
        # 由于使用了注重结果奖励方式，所以起始状态也有奖励，做为 G 的初始值
        G = dataModel.get_reward(start_state)   
        curr_s = start_state        # 把给定的起始状态作为当前状态
        t = 1                       # 折扣因子
        done = False                # 分幕结束标志
        while (done is False):      # 本幕循环
            # 根据当前状态和转移概率获得:下一个状态,奖励,是否到达终止状态
            next_s, r, done = dataModel.step(curr_s)   
            G += math.pow(gamma, t) * r
            t += 1
            curr_s = next_s
        # end while
        G_sum += G # 先暂时不计算平均值，而是简单地累加
    # end for
    V = G_sum / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return V

# 过程奖励
def Sampling2(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 定义最终的返回值，G 的平均数
    # 循环多幕
    for episode in tqdm.trange(episodes):
        # 由于使用了注重结果奖励方式，所以起始状态也有奖励，做为 G 的初始值
        G = dataModel.get_reward(start_state)   
        curr_s = start_state        # 把给定的起始状态作为当前状态
        t = 1                       # 折扣因子
        done = False                # 分幕结束标志
        while (done is False):      # 本幕循环
            # 根据当前状态和转移概率获得:下一个状态,奖励,是否到达终止状态
            next_s, r, done = dataModel.step(curr_s)   
            G += math.pow(gamma, t) * r
            t += 1
            curr_s = next_s
        # end while
        G_sum += G # 先暂时不计算平均值，而是简单地累加
    # end for
    V = G_sum / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return V

if __name__=="__main__":

    # dataModel = DataModel(R2)
    # for start_state in dataModel.S:    # 遍历状态集中的每个状态作为起始状态
    #     v_s = Sampling(dataModel, start_state, 2000, 1)
    #     print(start_state, v_s)
    # exit(0)




    Rs = {"状态奖励:值1":R1, "状态奖励:值2":R2, "过程奖励:值1":R3, "过程奖励:值2":R4}
    for name,R in Rs.items():
        print(str.format("{0}:{1}", name, R))
        dataModel = DataModel(R)
        gammas = [1, 0.9]
        for gamma in gammas:
            print("\tgamma =", gamma)
            v = Matrix(dataModel, gamma)
            print_V(v)
