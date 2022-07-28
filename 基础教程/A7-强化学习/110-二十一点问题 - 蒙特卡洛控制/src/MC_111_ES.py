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

def policy(state):
    if state[0] >= 20:
        return STICK
    else:
        return HIT


def run_V(env, gamma, episodes):

    Value = np.zeros((22, 11, 2))   # G 的总和
    Count = np.zeros((22, 11, 2))   # G 的数量

    for episode in tqdm.trange(episodes):
        s = env.reset()
        Episode = []     # 一幕内的(状态,动作,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            int_s = (s[0], s[1], int(s[2]))
            action = policy(s)
            next_s, reward, done, _ = env.step(action)
            Episode.append((int_s, action, reward))
            s = next_s  # 迭代

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1

    Count[Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
    V = Value / Count   # 求均值
    return V


def run_Q(env, gamma, episodes):

    Value = np.zeros((22, 11, 2))   # G 的总和
    Count = np.zeros((22, 11, 2))   # G 的数量

    for episode in tqdm.trange(episodes):
        s = env.reset()
        Episode = []     # 一幕内的(状态,动作,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            int_s = (s[0], s[1], int(s[2]))
            action = policy(s)
            next_s, reward, done, _ = env.step(action)
            Episode.append((int_s, action, reward))
            s = next_s  # 迭代

        num_step = len(Episode)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1

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