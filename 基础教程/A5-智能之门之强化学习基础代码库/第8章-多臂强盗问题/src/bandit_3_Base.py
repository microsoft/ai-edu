import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import trange
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False

class KArmBandit(object):
    def __init__(self, k_arms=10, mu=0, sigma=1): # 臂数,奖励分布均值,奖励分布方差
        self.k_arms = k_arms    # 臂数
        self.mu = mu            # 奖励均值
        self.sigma = sigma      # 奖励方差
        self.__best_arm = self.k_arms - 1   # 对算法透明，用于统计

    def reset(self):
        # 初始化 k 个 arm 的期望收益，并排序，但算法不要依赖这个排序
        self.E = np.sort(self.sigma * np.random.randn(self.k_arms) + self.mu)
        # 初始化 k 个 arm 的动作估值 Q_n 为 0
        self.Q = np.zeros(self.k_arms)
        # 保存每个 arm 被选择的次数 n
        self.action_count = np.zeros(self.k_arms, dtype=int)
        self.steps = 0   # 总步数，用于统计

    # 得到下一步的动作（下一步要使用哪个arm，由算法决定）
    def select_action(self):
        raise NotImplementedError

    # 执行指定的动作，并返回此次的奖励
    def pull_arm(self, action):
        reward =  np.random.randn() + self.E[action]
        return reward

    # 更新 q_n
    def update_Q(self, action, reward):
        # 总次数(time)
        self.steps += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        # 计算动作价值，采样平均
        self.Q[action] += (reward - self.Q[action]) / self.action_count[action]

    # 模拟运行
    def simulate(self, runs, steps):
        # 记录历史 reward，便于后面统计
        rewards = np.zeros(shape=(runs, steps))
        num_actions_per_arm = np.zeros(self.k_arms, dtype=int)
        is_best_action = np.zeros(shape=(runs, steps), dtype=int)
        for r in trange(runs):
            # 每次run都清零计算 q 用的统计数据，并重新初始化奖励均值
            self.reset()
            # 测试 time 次
            for s in range(steps):
                action = self.select_action()
                reward = self.pull_arm(action)
                self.update_Q(action, reward)
                rewards[r, s] = reward
                if (action == self.__best_arm): # 是否为最佳动作
                    is_best_action[r, s] = 1
            # end for t
            num_actions_per_arm += self.action_count    # 每个动作的选择次数
        # end for r
        return rewards, is_best_action, num_actions_per_arm
#end class

import multiprocessing as mp

# 多进程运行 simulate() 方法
def mp_simulate(bandits, k_arms, runs, steps, labels, title):
    all_rewards = []
    all_best = []
    all_actions = []
    print(labels)
    # 多进程执行
    pool = mp.Pool(processes=4)
    results = []
    for i, bandit in enumerate(bandits):
        results.append(pool.apply_async(bandit.simulate, args=(runs,steps,)))
    pool.close()
    pool.join()
    # 收集结果
    for i in range(len(results)):
        rewards, best_action, actions = results[i].get()
        all_rewards.append(rewards)
        all_best.append(best_action)
        all_actions.append(actions)
    # 计算统计数据
    all_best_actions = np.array(all_best).mean(axis=1)
    all_mean_rewards = np.array(all_rewards).mean(axis=1)
    all_done_actions = np.array(all_actions)
    # 最优动作选择的频率
    best_action_per_bandit = all_done_actions[:,k_arms-1]/all_done_actions.sum(axis=1)
    # 平均奖励值
    mean_reward_per_bandit = all_mean_rewards.sum(axis=1) / steps

    # 绘图
    lines = ["-", "--", "-.", ":"]  # 线条风格
    # 四行三列
    grid = plt.GridSpec(nrows=4, ncols=3)
    plt.figure(figsize=(15, 10))
    # 绘制average reward[0:100]
    plt.subplot(grid[0:2, 0])
    for i, mean_rewards in enumerate(all_mean_rewards):     
        tmp = ss.savgol_filter(mean_rewards[0:100], 10, 3)
        plt.plot(tmp, label=labels[i] + str.format("={0:0.3f}", mean_reward_per_bandit[i]), linestyle=lines[i])
    plt.ylabel('平均收益(0~100)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    # 绘制average reward[300:500]
    plt.subplot(grid[0:1, 1])
    for i, mean_rewards in enumerate(all_mean_rewards):
        tmp = ss.savgol_filter(mean_rewards[300:500], 20, 3)
        plt.plot(tmp, linestyle=lines[i])
    ticks = [0,50,100,150,200]
    tlabels = [300,350,400,450,500]
    plt.xticks(ticks, tlabels)
    plt.ylabel('平均收益(300~500)', fontsize=14)
    plt.grid()
    # 绘制average reward[700:900]
    plt.subplot(grid[1:2, 1])
    for i, mean_rewards in enumerate(all_mean_rewards):
        tmp = ss.savgol_filter(mean_rewards[700:900], 20, 3)
        plt.plot(tmp, linestyle=lines[i])
    ticks = [0,50,100,150,200]
    tlabels = [700,750,800,850,900]
    plt.xticks(ticks, tlabels)
    plt.ylabel('平均收益(700~900)', fontsize=14)
    plt.grid()
    # 绘制正确的最优动作选择频率
    plt.subplot(grid[2:4, 0:2])
    for i, counts in enumerate(all_best_actions):
        tmp = ss.savgol_filter(counts, 20, 3)
        plt.plot(tmp, label=labels[i] + str.format("={0:0.3f}", best_action_per_bandit[i]), linestyle=lines[i])
    plt.xlabel('迭代步数', fontsize=14)
    plt.ylabel('最佳动作采用比例', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    # 绘制所有动作的执行次数
    all_done_actions = (all_done_actions/runs + 0.5).astype(int)
    X = ["0","1","2","3","4","5","6","7","8","9"]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(4):
        ax = plt.subplot(grid[i, 2])
        Y = all_done_actions[i].tolist()
        ax.bar(X, Y, label=labels[i], color=colors[i])
        for x, y in zip(X, Y):   # 在bar上方标出动作执行次数
            ax.text(x, y, str(y), ha='center')
        ax.legend(fontsize=14)
    plt.show()

    return 

if __name__=="__main__":
    # 模拟运行
    runs = 2000
    steps = 1000
    k_arms = 10
    
    # 记录历史 reward，便于后面统计
    rewards = np.zeros(shape=(runs, steps))
    num_actions_per_arm = np.zeros(k_arms, dtype=int)
    is_best_action = np.zeros(shape=(runs, steps), dtype=int)
    np.random.seed(5)
    bandit = KArmBandit(k_arms)
    # 运行 2000 次取平均值
    for r in trange(runs):
        # 每次run都清空统计数据，但是使用相同的初始化参数
        bandit.reset()
        # 玩 1000 轮
        for t in range(steps):
            action = np.random.randint(k_arms)
            reward = bandit.pull_arm(action)
            bandit.update_Q(action, reward)
            rewards[r, t] = reward
            if (action == 9):
                is_best_action[r, t] = 1
        # end for t
        num_actions_per_arm += bandit.action_count
    # end for r
    # 平均收益
    r_m = rewards.mean(axis=0)
    smooth = ss.savgol_filter(r_m, 100, 3)
    plt.plot(smooth)
    plt.xlabel(u'训练步数')
    plt.ylabel(u'平均奖励')
    plt.grid()
    plt.show()
    # 动作选择次数
    X = ["0","1","2","3","4","5","6","7","8","9"]
    a_m = num_actions_per_arm / runs
    Y = a_m.tolist()
    plt.bar(X, Y)
    plt.xlabel(u'动作序号')
    plt.ylabel(u'动作选择次数')
    plt.show()
