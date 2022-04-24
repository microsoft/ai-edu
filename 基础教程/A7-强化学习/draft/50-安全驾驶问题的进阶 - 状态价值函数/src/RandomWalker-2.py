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

P = np.array(
    [  # S    A    B    C    D    H       
        [0.5, 0.5, 0,   0,   0  , 0  ], # S
        [0.5, 0,   0.5, 0,   0,   0, ], # A
        [0,   0.5, 0,   0.5, 0,   0, ], # B
        [0,   0,   0.5, 0,   0.5, 0, ], # C
        [0,   0,   0,   0.5, 0,   0.5,], # D
        [0,   0,   0,   0,   0,   0], # Home
])

# 过程奖励值
R1 = {
    States.Start: {States.Start: 0,    States.A:0},
    States.A:     {States.Start:0,     States.B:0},
    States.B:     {States.A:0,         States.C:0},
    States.C:     {States.B:0,         States.D:0},
    States.D:     {States.C:0,         States.Home:1},
    States.Home:  {States.Home:0},
}

R2 = {
    States.Start: {States.Start:-1,    States.A:-1},
    States.A:     {States.Start:-1,    States.B:-1},
    States.B:     {States.A:-1,        States.C:-1},
    States.C:     {States.B:-1,        States.D:-1},
    States.D:     {States.C:-1,        States.Home:0},
    States.Home:  {States.Home:0},
}

class DataModel(object):
    def __init__(self, R):
        self.P = P                          # 状态转移矩阵
        self.R = R                          # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)                # 状态数量
        self.end_states = [self.S.Home]      # 终止状态集

    # 判断给定状态是否为终止状态
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    def get_reward(self, s_curr, s_next):
        return self.R[s_curr][s_next]

    # 根据转移概率前进一步，返回（下一个状态、即时奖励、是否为终止）
    def get_next(self, s_curr):
        if self.is_end(s_curr):
            return None
        else:
            s_next = np.random.choice(self.S, p=self.P[s_curr.value])
            return s_next

def print_V(V):
    print(V)
    vv = np.around(V,3)
    for s in dataModel.S:
        print(str.format("\t\t{0}:\t{1}", s.name, vv[s.value]))

# 多次采样获得回报 G 的数学期望，即状态价值函数 V
def Sampling(dataModel, start_state, episodes, gamma):
    G_sum = 0  # 多幕 G 的和, 最后求平均值
    # 循环多幕
    for episode in tqdm.trange(episodes):
        s_curr = start_state # 把给定的起始状态作为当前状态
        G = 0           # 设置本幕的初始值 G=0
        t = 0           # 步数计数器
        while True:
            s_next = dataModel.get_next(s_curr)   # 得到下一个转移状态
            if (s_next is None):
                break
            r = dataModel.get_reward(s_curr, s_next) # 获得过程奖励值
            G += math.pow(gamma, t) * r # 当t=0时, G += r
            t += 1                      # 时间步+1
            s_curr = s_next
        # end while
        G_sum += G # 先暂时不计算平均值，而是简单地累加
    # end for
    V = G_sum / episodes   # 最后再一次性计算平均值，避免增加计算开销
    return V

R3 = [
    0,  # S
    0,  # A
    0,  # B
    0,  # C
    0.5,   # D
    0,                  # Home
]
R4 = [
    (-1)*0.5+(-1)*0.5,  # S
    (-1)*0.5+(-1)*0.5,  # A
    (-1)*0.5+(-1)*0.5,  # B
    (-1)*0.5+(-1)*0.5,  # C
    (-1)*0.5+(0)*0.5,  # D
    0*1.0,                 # Home
]

def Matrix(dataModel, gamma):
    I = np.eye(dataModel.N) * (1+1e-7)
    #I = np.eye(dataModel.N)
    tmp1 = I - gamma * dataModel.P
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, dataModel.R)
    return vs

if __name__=="__main__":
    episodes = 1000
    Rs = [R1, R2]
    #dataModel = DataModel(R4)
    #print("精确状态值:", Matrix(dataModel, 1))
    #print("精确状态值:", Matrix(dataModel, 0.9))
    for R in Rs:
        print("奖励值:", R)
        dataModel = DataModel(R)
        gammas = [1,0.9]
        for gamma in gammas:
            print("gamma =", gamma)
            for start_state in dataModel.S:    # 遍历状态集中的每个状态作为起始状态
                v_s = Sampling(dataModel, start_state, episodes, gamma)
                print(start_state, v_s)
