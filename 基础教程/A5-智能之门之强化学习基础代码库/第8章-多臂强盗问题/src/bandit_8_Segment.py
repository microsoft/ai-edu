
import multiprocessing as mp
import numpy as np

from bandit_3_Base import KArmBandit, mp_simulate
from bandit_4_Greedy import KAB_Greedy
from bandit_5_Softmax import KAB_Softmax
from bandit_6_UCB import KAB_UCB
from bandit_7_Thompson import KAB_Thompson


def staristic(k_arms, runs, steps):

    bandits:KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 40))
    bandits.append(KAB_Softmax(k_arms, 1.0))
    bandits.append(KAB_UCB(k_arms, 1.0))
    bandits.append(KAB_Thompson(k_arms, 0.7))

    # statistic
    all_rewards = []
    all_best = []
    all_actions = []

    pool = mp.Pool(processes=4)
    results = []
    for i, bandit in enumerate(bandits):
        results.append(pool.apply_async(bandit.simulate, args=(runs,steps,)))
    pool.close()
    pool.join()

    for i in range(len(results)):
        rewards, best_action, actions = results[i].get()
        all_rewards.append(rewards)
        all_best.append(best_action)
        all_actions.append(actions)

    all_best_actions = np.array(all_best).mean(axis=1)
    all_mean_rewards = np.array(all_rewards).mean(axis=1)
    all_done_actions = np.array(all_actions)
    best_action_per_bandit = all_done_actions[:,k_arms-1]/all_done_actions.sum(axis=1)
    mean_reward_per_bandit = all_mean_rewards.sum(axis=1) / steps

    features = np.zeros(shape=(len(bandits),8))
    # 0-100步的平均收益
    features[:,0] = all_mean_rewards[:,0:100].mean(axis=1)
    # 300-500步的平均收益
    features[:,1] = all_mean_rewards[:,300:500].mean(axis=1)
    # 700-900步的平均收益
    features[:,2] = all_mean_rewards[:,700:900].mean(axis=1)
    # 1000步的平均收益
    features[:,3] = mean_reward_per_bandit
    # 0-100步的最佳利用率
    features[:,4] = all_best_actions[:,0:100].mean(axis=1)
    # 300-500步的最佳利用率
    features[:,5] = all_best_actions[:,300:500].mean(axis=1)
    # 700-900步的最佳利用率
    features[:,6] = all_best_actions[:,700:900].mean(axis=1)
    # 1000步的最佳利用率
    features[:,7] = best_action_per_bandit


    print(np.round(features, 3))

    X = features
    # X: 第一维是不同的算法，第二维是8个特征值
    # 归一化, 按特征值归一化
    Y = (X - np.min(X, axis=0, keepdims=True)) / (np.max(X, axis=0, keepdims=True) - np.min(X, axis=0, keepdims=True))
    print("Y.shape=", Y.shape)
    print(np.round(Y, 3))

    # 计算权重值
    Z = Y / np.sqrt(np.sum(Y * Y))
    print("Z.shape=", Z.shape)
    print(np.round(Z, 3))

    # Z+  Z-
    max_z = np.max(Z, axis=0)
    min_z = np.min(Z, axis=0)
    print("max_z.shape=", max_z.shape)
    print(max_z)
    print(min_z)

    # D+, D-
    d_plus = np.sqrt(np.sum(np.square(Z - max_z), axis=1))
    d_minus = np.sqrt(np.sum(np.square(Z - min_z), axis=1))
    print("d_plus.shape=", d_plus.shape)
    print(d_plus)
    print(d_minus)

    C = d_minus / (d_plus + d_minus)
    print("C=", C)
    sort = np.argsort(C)
    print("sort.shape=",sort.shape)
    best_to_worst = list(reversed(sort))
    print(best_to_worst)
    for i in best_to_worst:
        print(bandits[i].__class__.__name__)


if __name__ == "__main__":
    runs = 2000
    steps = 1000
    k_arms = 10

    np.random.seed(515)
    bandits:KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 40))
    bandits.append(KAB_Softmax(k_arms, 1.0))
    bandits.append(KAB_UCB(k_arms, 1.0))
    bandits.append(KAB_Thompson(k_arms, 0.7))

    labels = [
        'Greedy   (40)',
        'Softmax (1.0)',
        'UCBound (1.0)',
        'Thompson(0.7)',
    ]
    title = "Compare"
    mp_simulate(bandits, k_arms, runs, steps, labels, title)

#   staristic(k_arms, runs, steps)
