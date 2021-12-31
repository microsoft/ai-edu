import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

class KArmBandit(object):
    def __init__(self, k_arms=10):
        self.k_arms = k_arms

    def reset(self):
        # 初始化一个 k 个元素的正态分布数组，均值为0，方差为1，
        # 作为 k 个 arm 的基本收益均值，当mu=0时有可能是正或负
        mu = 0
        sigma = 1
        # sort from small to big, so the 10th is the best
        self.q_base = np.sort(sigma * np.random.randn(self.k_arms) + mu)
        
        assert(np.argmax(self.q_base) == self.k_arms-1)
        
        # 初始化 k 个 arm 的动作估值q*为 0
        self.Q = np.zeros(self.k_arms)
        # 保存每个 arm 被选择的次数
        self.action_count = np.zeros(self.k_arms, dtype=int)
        self.step = 0
        # 每个arm上执行到目前为止的平均收益(总收益/总被选中次数)
        

    # 得到下一步的动作（下一步要使用哪个arm）
    def select_action(self):
        pass


    # 执行指定的动作，并返回此次的奖励
    def step_reward(self, action):
        # sigma = 1
        # mu = self.q_base[action]
        # 方差为1，均值为该 arm 的 q_base 对应的初始化值
        # reward =  sigma * np.random.randn() + mu
        reward =  np.random.randn() + self.q_base[action]
        return reward


    # 更新 q*
    def update_Q(self, action, reward):
        # 总次数(time)
        self.step += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        # 计算动作价值，采样平均
        self.Q[action] += (reward - self.Q[action]) / self.action_count[action]


    # 模拟运行
    def simulate(self, runs, steps):
        # 记录历史 reward，便于后面统计
        rewards = np.zeros(shape=(runs, steps))
        actions = np.zeros(self.k_arms, dtype=int)
        best_action = np.zeros(shape=(runs, steps), dtype=int)
        for r in trange(runs):
            # 每次run都独立，但是使用相同的参数
            self.reset()
            # 测试 time 次
            for s in range(steps):
                action = self.select_action()
                reward = self.step_reward(action)
                self.update_Q(action, reward)
                rewards[r, s] = reward
                if (action == self.k_arms-1):
                    best_action[r, s] = 1
            # end for t
            actions += self.action_count
        # end for r
        return rewards, best_action, actions



import multiprocessing as mp

def mp_simulate(bandits, k_arms, runs, steps, labels):

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
        print(labels[i])
        all_rewards.append(rewards)
        all_best.append(best_action)
        all_actions.append(actions)

    all_best_actions = np.array(all_best).mean(axis=1)
    all_mean_rewards = np.array(all_rewards).mean(axis=1)
    all_done_actions = np.array(all_actions)
    best_action_per_bandit = all_done_actions[:,k_arms-1]/all_done_actions.sum(axis=1)
    mean_reward_per_bandit = all_mean_rewards.sum(axis=1) / steps

    # draw

    #grid = plt.GridSpec(nrows=4, ncols=3, wspace=0.2, hspace=0.2)
    grid = plt.GridSpec(nrows=4, ncols=3)
    plt.figure(figsize=(15, 20))

    plt.subplot(grid[0:2, 0])
    for i, mean_rewards in enumerate(all_mean_rewards):
        plt.plot(mean_rewards[0:100], label=labels[i] + str.format("{0:0.4f}", mean_reward_per_bandit[i]))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.subplot(grid[0:2, 1])
    for i, mean_rewards in enumerate(all_mean_rewards):
        x = mean_rewards[500:1000]
        plt.plot(x, label=labels[i] + str.format("{0:0.4f}", mean_reward_per_bandit[i]))
    ticks = [0,100,200,300,400,500]
    tlabels = [500,600,700,800,900,1000]
    plt.xticks(ticks, tlabels)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()
    

    plt.subplot(grid[2:4, 0:2])
    for i, counts in enumerate(all_best_actions):
        plt.plot(counts, label=labels[i] + str.format("{0:0.3f}", best_action_per_bandit[i]))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.grid()
    
    X = ["0","1","2","3","4","5","6","7","8","9"]
    for i in range(4):
        plt.subplot(grid[i, 2])
        Y = all_done_actions[i].tolist()
        plt.bar(X, Y, label=labels[i])
        for x,y in zip(X, Y):
            plt.text(x,y, str(y), ha='center')
        plt.legend()


    plt.show()

    return 

    plt.figure(figsize=(10, 20))
    # 左上角，绘制step从0到200
    plt.subplot(2, 2, 1)
    for i, mean_rewards in enumerate(all_mean_rewards):
        plt.plot(mean_rewards[0:200], label=labels[i] + str.format("{0:0.4f}", mean_reward_per_bandit[i]))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()
    # 右上角，绘制step从800到100
    plt.subplot(2, 2, 2)
    for i, mean_rewards in enumerate(all_mean_rewards):
        plt.plot(mean_rewards[800:1000], label=labels[i] + str.format("{0:0.4f}", mean_reward_per_bandit[i]))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for i, counts in enumerate(all_best_actions):
        plt.plot(counts, label=labels[i] + str.format("{0:0.3f}", best_action_per_bandit[i]))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.grid()

    #plt.subplot(2, 1, 2)
    #plt.barh(["0","1","2","3","4","5","6","7","8","9"], all_done_actions.tolist())
    #plt.barh(["0","1","2"], [1,2,3])
    plt.show()
