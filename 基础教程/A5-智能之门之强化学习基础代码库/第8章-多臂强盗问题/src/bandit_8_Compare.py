
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np

from bandit_3_Base import KArmBandit
from bandit_4_Greedy import KAB_Greedy
from bandit_4_E_Greedy import KAB_E_Greedy
from bandit_5_Softmax import KAB_Softmax
from bandit_6_UCB import KAB_UCB
from bandit_7_Thompson import KAB_Thompson


def run_algo(algo_name:KArmBandit, runs, steps, k_arms, parameters):
    all_mean_reward = []
    for p in parameters:
        bandit = algo_name(k_arms, p)
        rewards, _, _ = bandit.simulate(runs, steps)
        mean_reward = rewards.mean()
        all_mean_reward.append(mean_reward)
    return all_mean_reward
        

if __name__ == "__main__":
    runs = 200
    steps = 1000
    k_arms = 10

    algo_names = [KAB_Greedy, KAB_E_Greedy, KAB_Softmax, KAB_UCB, KAB_Thompson]
    algo_params = {
        KAB_Greedy: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        KAB_E_Greedy: [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
        KAB_Softmax: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        KAB_UCB: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2],
        KAB_Thompson: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }

    np.random.seed(5)

    pool = mp.Pool(processes=4)
    results = []
    for algo_name in algo_names:
        params = algo_params[algo_name]
        results.append(pool.apply_async(run_algo, args=(algo_name,runs,steps,k_arms,params,)))
    pool.close()
    pool.join()
    # 收集结果
    algo_rewards = []
    for i in range(len(results)):
        algo_reward = results[i].get()
        algo_rewards.append(algo_reward)
    print(len(algo_rewards))
    markers = ['s', 'x', 'o', "^", "v"]
    for i, algo_name in enumerate(algo_names):
        smooth_data = ss.savgol_filter(algo_rewards[i], 10, 3)
        plt.plot(smooth_data, label=str(algo_name), marker=markers[i])
        print(algo_name)
        print(algo_rewards[i])
    plt.grid()
    plt.legend()
    plt.xlabel = "参数"
    plt.show()

