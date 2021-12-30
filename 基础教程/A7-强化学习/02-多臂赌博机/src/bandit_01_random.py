import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean

from tqdm import trange
from bandit_00_base import *


class K_ArmBandit_01_Random(K_ArmBandit_0):
    # 得到下一步的动作（下一步要使用哪个arm）
    def select_action(self):
        # 从 k 个 arm 里随机选一个
        action = np.random.randint(low=0, high=self.k_arms)
        return action


if __name__ == "__main__":

    runs = 2000
    steps = 1000
    k_arms = 3

    # shape=(bandit, runs, steps, 2[action:reward])
    all_history = []

    bandits = []
    bandits.append(K_ArmBandit_01_Random(3, [0.3, 0.5, 0.8]))


    all_summary = []
    for bandit in bandits:
        bandit.simulate(runs, steps)
        s1, s2 = bandit.summary()
        np.set_printoptions(suppress=True)
        print(np.round(np.array(s1), 3))
        all_summary.append(s2)

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)

    for mean_reward in np.array(all_summary)[:,0]:
        plt.plot(mean_reward)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for best_action in np.array(all_summary)[:,1]:
        plt.plot(best_action)
    plt.xlabel('steps')
    plt.ylabel('best action')
    plt.legend()
    plt.grid()

    plt.show()

