
from enum import Enum
import numpy as np
import math

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


def print_V(V, round, shape, seperator: SeperatorLines, info: str):
    np.set_printoptions(suppress=True)
    print_seperator_line(seperator, info)
    V = np.around(V, round)
    V = np.reshape(V, shape)
    print(V)

def print_Policy(policy, round, shape, seperator: SeperatorLines, info: str):
    print_Q(policy, round, shape, seperator, info)

def print_Q(Q, round, shape, seperator: SeperatorLines, info: str):
    print_seperator_line(seperator, info)
    a = Q.tolist()
    for i in range(shape[0]):
        for j in range(shape[1]):
            list_round = [np.round(i, round) for i in a[i*shape[1]+j]]
            print(list_round, end="\t")
        print("")

def RMSE(x, y):
    err = np.sqrt(np.sum(np.square(x - y))/y.shape[0])
    return err

# 二范数的误差商，0-无穷大
def Norm2Err(x, ground_truth):
    a = np.linalg.norm(x - ground_truth, 2)
    b = np.linalg.norm(ground_truth, 2)
    return a/b

# 二范数的误差商，缩放到 0-1
def Norm2Err01(x, ground_truth):
    a = np.linalg.norm(x - ground_truth, 2)
    b = np.linalg.norm(ground_truth, 2)
    err = a/b
    y = err / (err + math.exp(-err))
    return y


def calculate_angle_between_two_vectors(x, y):
    a = np.dot(x, y)
    b = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)
    theta = math.acos(a/b) * 180 / math.pi
    return theta
    

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


# 软策略转硬策略
def Soft2Hard(soft_policy):
    hard_policy = np.zeros_like(soft_policy)
    for s in range(len(soft_policy)):
        best_actions = np.argwhere(soft_policy[s] == np.max(soft_policy[s]))
        best_actions_count = len(best_actions)
        hard_policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(len(soft_policy[s]))]
        # a = np.argmax(soft_policy[s])
        # hard_policy[s, a] = 1
    return hard_policy


def Hard2Soft(hard_policy, epsilon=0.3):
    soft_policy = np.zeros_like(hard_policy)
    for s in range(len(hard_policy)):
        soft_policy[s] = epsilon / len(hard_policy[s])
        a = np.argmax(hard_policy[s])
        soft_policy[s, a] += 1 - epsilon
    return soft_policy

# x = np.array([1,2,3])
# y = np.array([1.1,2,2.9])
# print(RMSE(x, y))
# print(RMSE(x/10, y/10))
# print(Norm2Err(x, y))
# print(Norm2Err(x/10, y/10))
