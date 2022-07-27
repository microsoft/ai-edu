import numpy as np
import gym
import Algorithm.Base_MC_Policy_Iteration as base
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

HIT = 1
STICK = 0

def policy(state):
    if state[0] >= 20:
        return STICK
    else:
        return HIT


def run(env, gamma):

    Value = np.zeros((22, 11, 2))   # G 的总和
    Count = np.zeros((22, 11, 2))   # G 的数量

    for episode in tqdm.trange(50000):
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
    env = gym.make('Blackjack-v1', sab=True)
    V = run(env, gamma)
    env.close()
    #print(V)

    fig = sns.heatmap(np.flipud(V[12:,1:,0]), cmap="YlGnBu", xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
    fig.set_ylabel('player sum', fontsize=30)
    fig.set_xlabel('dealer showing', fontsize=30)

    plt.show()

