import numpy as np
from enum import Enum

class SimpleModel(object):
    def __init__(self, States, P, R1 = None, R2 = None, end_states=None):
        self.P = P              # 状态转移矩阵
        self.R1 = R1            # 过程奖励，如果存在，则R2=None
        self.R2 = R2            # 期望奖励，如果存在，则R1=None
        self.S = States         # 状态集
        self.nS = len(self.S)   # 状态数量
        if end_states is None:
            self.end_states = [list(States)[-1]]     # 终止状态集
        else:
            self.end_states = end_states

    # 判断给定状态是否为终止状态
    def is_end(self, s):
        if (s in self.end_states):
            return True
        return False

    # 过程奖励
    def get_reward_1(self, s, s_next):
        assert self.R1 is not None
        return self.R1[s][s_next]

    # 期望奖励
    def get_reward_2(self, s):
        assert self.R2 is not None
        return self.R2[s.value]

    # 根据转移概率前进一步，返回（下一个状态、即时奖励、是否为终止）
    def step(self, s):
        next_s = np.random.choice(self.S, p=self.P[s.value])
        if self.R1 is not None:
            reward = self.get_reward_1(s, next_s)
        else:
            reward = self.get_reward_2(s)
        is_done = self.is_end(next_s)
        return next_s, reward, is_done


def Matrix(dataModel, gamma):
    I = np.eye(dataModel.nS) * (1+1e-7)
    #I = np.eye(dataModel.nS)
    tmp1 = I - gamma * dataModel.P
    tmp2 = np.linalg.inv(tmp1)
    vs = np.dot(tmp2, dataModel.R)
    return vs

# R2 = [-1, -1, -1, -1, -1, 0]
# R1 = [ 0,  0,  0,  0,  0, 1]
# dm = StateBased_DataModel(R1)
# print(Matrix(dm, 1.0))
# print(Matrix(dm, 0.9))

