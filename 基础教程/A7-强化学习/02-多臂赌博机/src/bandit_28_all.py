
from bandit_22_greedy import *
from bandit_23_e_greedy import *
from bandit_24_optimistic_initial import *
from bandit_25_softmax import *
from bandit_26_ucb import *
from bandit_27_thompson import *


def staristic(k_arms, runs, steps):

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 25))
    bandits.append(KAB_E_Greedy(k_arms, 0.1))
    bandits.append(KAB_Optimistic_Initial(k_arms, 0.1, 5))
    bandits.append(KAB_Softmax(k_arms, 0.15, 0.8))
    bandits.append(KAB_UCB(k_arms, 1))
    bandits.append(KAB_Thompson(k_arms, 0.5))

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


    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 25))
    bandits.append(KAB_E_Greedy(k_arms, 0.1))
    bandits.append(KAB_Optimistic_Initial(k_arms, 0.1, 5))
    bandits.append(KAB_Softmax(k_arms, 0.15, 0.8))

    labels = [
        'Greedy(25), ',
        'E_Greedy(0.1), ',
        'Optimistic(0.3,2), ',
        'Softmax(0.2,T,3), ',
    ]
    title = "Compare-1"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 25))
    bandits.append(KAB_Softmax(k_arms, 0.15, 0.8))
    bandits.append(KAB_UCB(k_arms, 1))
    bandits.append(KAB_Thompson(k_arms, 0.5))

    labels = [
        'Greedy(25), ',
        'Softmax(0.2,T,3), ',
        'UCB(1), ',
        'Thompson(0.5), ',
    ]
    title = "Compare-2"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)


    staristic(k_arms, runs, steps)
