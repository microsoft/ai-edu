
from enum import Enum
import numpy as np


class SeperatorLines(Enum):
    empty = 0   # 打印空行
    short = 1   # 打印10个'-'
    middle = 2  # 打印30个'-'
    long = 3    # 打印40个'='

def print_seperator_line(style: SeperatorLines, info=None):
    if style == SeperatorLines.empty:
        print("")
    elif style == SeperatorLines.short:
        if (info is None):
            print("-"*10)
        else:
            print("----- " + info + " -----")
    elif style == SeperatorLines.middle:
        if (info is None):
            print("-"*30)
        else:
            print("-"*15 + info + "-"*15)
    elif style == SeperatorLines.long:
        if (info is None):
            print("="*40)
        else:
            print("="*20 + info + "="*20)


def print_V(dataModel, V):
    vv = np.around(V,2)
    print("状态价值函数计算结果(数组) :", vv)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))


def RMSE(x, y):
    err = np.sqrt(np.sum(np.square(x - y))/y.shape[0])
    return err

def test_policy(env, policy, episodes=100):
    R = 0
    for i in range(episodes):
        s = env.reset()
        done = False
        while (done is False):            # 幕内循环
            action = np.argmax(policy[s])
            # action = np.random.choice(env.action_space.n, p=policy[s])
            next_s, reward, done, info = env.step(action)
            R += reward
            if done == True and reward == 0:
                print(s, action, next_s)
            s = next_s

    return R

# 根据输入的4个概率值创建策略
def create_policy(nS, nA, args):
    assert(nA == len(args))
    left = args[0]
    down = args[1]
    right = args[2]
    up = args[3]
    assert(left+down+right+up==1)
    policy = np.zeros((nS, nA))
    policy[:, 0] = left
    policy[:, 1] = down
    policy[:, 2] = right
    policy[:, 3] = up
    return policy

# 从Q函数表格中抽取策略
def extract_policy_from_Q(Q, end_states):
    policy = np.zeros_like(Q)
    for s in range(Q.shape[0]):
        if s not in end_states:
            max_v = np.max(Q[s])
            for a in range(Q[s].shape[0]):
                if Q[s,a] == max_v:
                    policy[s, a] = 1
    return policy
