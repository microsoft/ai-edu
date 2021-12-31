from functools import update_wrapper
import re
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import trange
from bandit_00_base import *
from bandit_01_random import *
from bandit_02_greedy import *
from bandit_03_e_greedy import *
from bandit_04_softmax import *

class K_ArmBandit_05_Thompson(K_ArmBandit_0):
    def reset(self, steps):
        super().reset(steps)
        self.succ = np.ones(self.k_arms)
        self.fail = np.ones(self.k_arms)


    # 得到下一步的动作（下一步要使用哪个arm）
    def select_action(self):
        beta = np.random.beta(self.succ, self.fail)
        action = np.argmax(beta)
        return action

    def update_counter(self, action, reward):
        super().update_counter(action, reward)
        if (reward == 0):
            self.fail[action] += 1
        else:
            self.succ[action] += 1

if __name__ == "__main__":

    runs = 1000
    steps = 1000
    k_arms = 3

    # shape=(bandit, runs, steps, 2[action:reward])
    all_history = []

    bandits = []
    bandits.append(K_ArmBandit_05_Thompson(3, [0.3, 0.5, 0.8]))
    bandits.append(K_ArmBandit_04_Softmax(3, [0.3, 0.5, 0.8], alpha=0.1, has_base=True))    
    bandits.append(K_ArmBandit_02_Greedy(3, [0.3, 0.5, 0.8], search_step=20))
    bandits.append(K_ArmBandit_03_Eps_Greedy(3, [0.3, 0.5, 0.8], epsilon=0.1))
    

    labels = [r'thompson, ',
              r'softmax, 0.1, False, ',
              r'greedy, 20, ',
              r'e-greedy, 0.1, ']

    all_summary = []
    all_mean = []
    np.set_printoptions(suppress=True)

    
    pool = mp.Pool(processes=4)
    results = [None]*len(bandits)
    for i, bandit in enumerate(bandits):
        results[i] = pool.apply_async(bandit.simulate, args=(runs,steps,))
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

