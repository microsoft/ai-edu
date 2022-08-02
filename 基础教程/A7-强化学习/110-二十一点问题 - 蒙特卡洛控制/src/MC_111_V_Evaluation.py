import numpy as np
import gym
import Algorithm.Base_MC_Policy_Iteration as base
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False


HIT = 1
STICK = 0

def static_policy(state):
    if state[0] >= 20:
        return STICK
    else:
        return HIT


def run_V(env, gamma, episodes):
    # 玩家手牌点数和：1-21, 单元 0 空置
    # 庄家明牌点数：  1-10, 单元 0 空置
    # 有无可用的A：   0-1 （无/有）
    Value = np.zeros((22, 11, 2))   # G 的总和
    Count = np.zeros((22, 11, 2))   # G 的数量

    for episode in tqdm.trange(episodes):
        s = env.reset()
        Episode = []     # 一幕内的(状态,动作,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            int_s = (s[0], s[1], int(s[2]))
            action = static_policy(s)
            next_s, reward, done, _ = env.step(action)
            Episode.append((int_s, reward))
            s = next_s  # 迭代

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, r = Episode[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1

    print(Count)
    print(np.min(Count[12:,1:,]))

    Count[Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
    V = Value / Count   # 求均值
    return V

def greedy_policy(state, Q):
    max_A = np.max(Q[state])    # 从几个动作中取最大值
    argmax_A = np.where(Q[state] == max_A)[0]
    action = np.random.choice(argmax_A)
    return action

def run_Q(env, gamma, episodes, policy):
    # 玩家手牌点数和：1-21, 单元 0 空置
    # 庄家明牌点数：  1-10, 单元 0 空置
    # 有无可用的A：   0-1 （无/有）
    # 动作：0-停牌，1-要牌
    Value = np.zeros((22, 11, 2, 2))   # G 的总和
    Count = np.ones((22, 11, 2, 2))   # G 的数量
    Q = Value / Count

    for episode in tqdm.trange(episodes):
        s = env.reset()
        Episode = []     # 一幕内的(状态,动作,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            int_s = (s[0], s[1], int(s[2]))
            action = greedy_policy(s, Q)
            next_s, reward, done, _ = env.step(action)
            Episode.append((int_s, action, reward))
            s = next_s  # 迭代

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s][a] += G     # 值累加
            Count[s][a] += 1     # 数量加 1

        # refine policy
        max_A = np.max(Q[state])    # 从几个动作中取最大值
        argmax_A = np.where(Q[state] == max_A)[0]
        action = np.random.choice(argmax_A)


    print(Count)
    print(np.min(Count[12:,1:,]))
    Count[Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
    V = Value / Count   # 求均值
    return V


if __name__=="__main__":
    gamma = 1
    episodes = 50000
    env = gym.make('Blackjack-v1', sab=True)
    V = run_V(env, gamma, episodes)
    env.close()
    #print(V)

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121)
    sns.heatmap(np.flipud(V[12:,1:,0]), cmap="Wistia", xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
    ax.set_ylabel('玩家的手牌', fontsize=16)
    ax.set_xlabel('庄家的明牌', fontsize=16)
    ax.set_title('无可用的 A')

    ax = fig.add_subplot(122)
    sns.heatmap(np.flipud(V[12:,1:,1]), cmap="Wistia", xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
    ax.set_ylabel('玩家的手牌', fontsize=16)
    ax.set_xlabel('庄家的明牌', fontsize=16)
    ax.set_title('有可用的 A')

    plt.show()


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zticks(np.arange(-1,1,1))
    X, Y = np.meshgrid(range(0,10), range(0,10))
    ax.plot_surface(X, Y, V[12:,1:,0]) 
    plt.show()