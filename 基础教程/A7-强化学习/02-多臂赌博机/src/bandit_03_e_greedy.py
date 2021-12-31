import re
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import trange
from bandit_00_base import *
from bandit_01_random import *
from bandit_02_greedy import *

class K_ArmBandit_03_Eps_Greedy(K_ArmBandit_02_Greedy):
    def __init__(self, k_arms=3, prob_list=[0.3, 0.5, 0.8], epsilon=0.0):
        super().__init__(k_arms=k_arms, prob_list=prob_list)
        self.epsilon = epsilon

    # 得到下一步的动作（下一步要使用哪个arm）
    def select_action(self):
        if (np.random.rand() < self.epsilon):
            action = np.random.randint(low=0, high=self.k_arms)
        else:
            action = np.argmax(self.average_reward)
        return action


if __name__ == "__main__":

    runs = 1000
    steps = 500
    k_arms = 3

    # shape=(bandit, runs, steps, 2[action:reward])
    all_history = []

    bandits = []
    #bandits.append(K_ArmBandit_01_Random(3, [0.3, 0.5, 0.8]))
    #bandits.append(K_ArmBandit_01_Greedy(3, [0.3, 0.5, 0.8], search_step=20))
    bandits.append(K_ArmBandit_03_Eps_Greedy(3, [0.3, 0.5, 0.8], epsilon=0.05))
    bandits.append(K_ArmBandit_03_Eps_Greedy(3, [0.3, 0.5, 0.8], epsilon=0.075))
    bandits.append(K_ArmBandit_03_Eps_Greedy(3, [0.3, 0.5, 0.8], epsilon=0.1))
    bandits.append(K_ArmBandit_03_Eps_Greedy(3, [0.3, 0.5, 0.8], epsilon=0.15))

    labels = [r'e-greedy, 0.05, ',
              r'e-greedy, 0.075, ',
              r'e-greedy, 0.1, ',
              r'e-greedy, 0.15, ']

    all_summary = []
    all_mean = []
    np.set_printoptions(suppress=True)

    
    pool = mp.Pool(processes=4)
    results = []
    for i, bandit in enumerate(bandits):
        results.append(pool.apply_async(bandit.simulate, args=(runs,steps,)))
    pool.close()
    pool.join()

    for i in range(len(results)):
        s1, s2, s3 = results[i].get()
        print(labels[i])
        print(np.round(np.array(s1), 3))
        all_summary.append(s2)
        all_mean.append(s3)

    '''
    for i, bandit in enumerate(bandits):
        bandit.simulate(runs, steps)
        s1, s2, s3 = bandit.summary()
        
        print(labels[i])
        print(np.round(np.array(s1), 3))
        all_summary.append(s2)
        all_mean.append(s3)
    '''

    summary_array = np.array(all_summary)

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)

    for i in range(summary_array.shape[0]):
        plt.plot(summary_array[i,0], label=labels[i] + str(all_mean[i]))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for i in range(summary_array.shape[0]):
        plt.plot(summary_array[i,1], label=labels[i])
    plt.xlabel('steps')
    plt.ylabel('% best action')
    plt.legend()
    plt.grid()

    plt.show()

