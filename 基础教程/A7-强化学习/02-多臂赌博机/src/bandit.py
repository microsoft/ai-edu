import re
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

class K_ArmBandit(object):
    def __init__(self, k, epsilon):
        self.k_arms = k
        self.epsilon = epsilon
        # 初始化一个 k 个元素的正态分布数组，均值为0，方差为1，
        # 作为 k 个 arm 的基本收益均值，有可能是正或负
        mu = 0
        sigma = 1
        self.q_base = sigma * np.random.randn(self.k_arms) + mu
        # 初始化 k 个arm的动作估值q*为 0
        self.q_star = np.zeros(self.k_arms)
        self.action_count = np.zeros(self.k_arms)
        self.time = 0
        self.action_idx = np.arange(self.k_arms)

    # 得到下一步的动作（下一步要使用哪个arm）
    def select_action(self):
        # 小于epsilon, 执行随机探索行动
        if (np.random.rand() < self.epsilon):
            return np.random.choice(self.action_idx)

        # 获得目前可以估计的最大值
        q_best_value = np.max(self.q_star)
        # 可能同时有多个最大值，获得它们在数组中的索引位置
        best_actions = np.where(q_best_value == self.q_star)[0]
        # 从多个最大值（索引）中随机选择一个作为下一步的动作
        action = np.random.choice(best_actions)
        return action

    # 执行指定的动作，并返回此次的奖励
    def step_reward(self, action):
        sigma = 1
        mu = self.q_base[action]
        reward =  sigma * np.random.randn() + mu
        self.update_q(action, reward)
        return reward

    # 更新 q*
    def update_q(self, action, reward):
        # 总次数(time)
        self.time += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        # 计算动作价值，采样平均
        self.q_star[action] = self.q_star[action] + (reward - self.q_star[action]) / self.action_count[action]

if __name__ == "__main__":

    epsilons = [0.01, 0.05, 0.1]
    runs = 2000
    time = 1000
    k_arms = 10


    total_action_count = np.zeros(k_arms)
    total_q_estimation = np.zeros(k_arms)

    rewards = np.zeros((len(epsilons), runs, time))
    for i, eps in enumerate(epsilons):
        # kab = K_ArmBandit(k_arms, eps)
        # run 2000 次后，取平均
        for r in trange(runs):
            kab = K_ArmBandit(k_arms, eps)
            # 测试 time 次
            for t in range(time):
                action = kab.select_action()
                reward = kab.step_reward(action)
                rewards[i, r, t] = reward
            # end for t
            total_action_count += kab.action_count
            total_q_estimation += kab.q_star

    print(total_action_count)
    print(total_q_estimation)

    # 取2000次的平均
    mean_rewards = rewards.mean(axis=1)
    for eps, mean_rewards in zip(epsilons, mean_rewards):
        plt.plot(mean_rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()
    plt.show()
