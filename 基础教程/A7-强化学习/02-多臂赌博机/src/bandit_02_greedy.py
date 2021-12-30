import re
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from bandit_00_base import *
from bandit_01_random import *

class K_ArmBandit_01_Greedy(K_ArmBandit_0):
    def __init__(self, k_arms=3, prob_list=[0.3, 0.5, 0.8], search_step=10):
        super().__init__(k_arms=k_arms, prob_list=prob_list)
        self.search_steps = search_step

    def reset(self, steps):
        super().reset(steps)
        self.average_reward = np.zeros(self.k_arms)

    # 得到下一步的动作（下一步要使用哪个arm）
    def select_action(self):
        if (self.step < self.search_steps):
            action = np.random.randint(low=0, high=self.k_arms)
        else:
            action = np.argmax(self.average_reward)
        return action

    def update_counter(self, action, reward):
        super().update_counter(action, reward)
        self.average_reward[action] += (reward - self.average_reward[action]) / self.action_count[action]

if __name__ == "__main__":

    runs = 1000
    steps = 1000
    k_arms = 3

    # shape=(bandit, runs, steps, 2[action:reward])
    all_history = []

    bandits = []
    bandits.append(K_ArmBandit_01_Random(3, [0.3, 0.5, 0.8]))
    bandits.append(K_ArmBandit_01_Greedy(3, [0.3, 0.5, 0.8], search_step=10))
    bandits.append(K_ArmBandit_01_Greedy(3, [0.3, 0.5, 0.8], search_step=20))
    bandits.append(K_ArmBandit_01_Greedy(3, [0.3, 0.5, 0.8], search_step=30))

    labels = [r'random, ',
              r'greedy, 10, ',
              r'greedy, 20, ',
              r'greedy, 30, ']

    all_summary = []
    all_mean = []
    np.set_printoptions(suppress=True)
    for i, bandit in enumerate(bandits):
        bandit.simulate(runs, steps)
        s1, s2, s3 = bandit.summary()
        
        print(labels[i])
        print(np.round(np.array(s1), 3))
        all_summary.append(s2)
        all_mean.append(s3)

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

