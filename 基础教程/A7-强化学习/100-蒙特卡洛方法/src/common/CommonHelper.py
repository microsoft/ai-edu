
from enum import Enum
import numpy as np


class SeperatorLines(Enum):
    empty = 0   # 打印空行
    short = 1   # 打印10个'-'
    middle = 2  # 打印30个'-'
    long = 3    # 打印40个'='

def print_seperator_line(style: SeperatorLines):
    if style == SeperatorLines.empty:
        print("")
    elif style == SeperatorLines.short:
        print("-"*10)
    elif style == SeperatorLines.middle:
        print("-"*30)
    elif style == SeperatorLines.long:
        print("="*40)


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
