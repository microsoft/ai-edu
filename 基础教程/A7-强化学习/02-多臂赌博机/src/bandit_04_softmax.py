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

class K_ArmBandit_04_Softmax(K_ArmBandit_0):
    def __init__(self, k_arms=3, prob_list=[0.3, 0.5, 0.8], alpha=0.1, has_base=False):
        super().__init__(k_arms=k_arms, prob_list=prob_list)
        self.alpha = alpha
        self.has_base = has_base

    def reset(self, steps):
        super().reset(steps)
        self.q_star = np.zeros(self.k_arms)


    # 得到下一步的动作（下一步要使用哪个arm）
    def select_action(self):
        tmp = np.exp(self.q_star - np.max(self.q_star))
        self.softmax = tmp / np.sum(tmp)
        action = np.random.choice(self.k_arms, p=self.softmax)
        return action


    def update_counter(self, action, reward):
        super().update_counter(action, reward)
        one_hot = np.zeros(self.k_arms)
        one_hot[action] = 1
        if (self.has_base):
            self.q_star += self.alpha * (reward - self.average_reward) * (one_hot - self.softmax)
        else:
            self.q_star += self.alpha * (reward) * (one_hot - self.softmax)

if __name__ == "__main__":

    runs = 1000
    steps = 1000
    k_arms = 3

    # shape=(bandit, runs, steps, 2[action:reward])
    all_history = []

    bandits = []
    bandits.append(K_ArmBandit_04_Softmax(3, [0.3, 0.5, 0.8], alpha=0.1, has_base=True))
    bandits.append(K_ArmBandit_04_Softmax(3, [0.3, 0.5, 0.8], alpha=0.1, has_base=False))
    #bandits.append(K_ArmBandit_01_Greedy(3, [0.3, 0.5, 0.8], search_step=20))
    #bandits.append(K_ArmBandit_01_Eps_Greedy(3, [0.3, 0.5, 0.8], epsilon=0.1))
    

    labels = [r'softmax, 0.1, True, ',
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

