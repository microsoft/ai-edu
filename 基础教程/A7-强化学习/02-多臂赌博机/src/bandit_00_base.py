import re
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

class K_ArmBandit_0(object):
    def __init__(self, k_arms=3, prob_list=[0.3, 0.5, 0.8]):
        assert(k_arms == len(prob_list))
        self.k_arms = k_arms
        self.prob = np.array(prob_list)

    def reset(self, steps):
        # 保存每个 step 被选择的 arm 的次数(index 0) 和 reward 值(index 1)
        self.action_reward = np.zeros(shape=(steps, 2))
        self.steps = steps
        self.step = 0
        self.action_count = np.zeros(self.k_arms)


    # 得到下一步的动作(需要子类来实现)
    def select_action(self):
        pass


    # 执行指定的动作，并返回此次的奖励
    def step_reward(self, action):
        if (np.random.rand() < self.prob[action]):
            return 1
        else:
            return 0
        '''
        reward = np.random.choice(2, p=self.prob[:, action])
        return reward
        '''

    def update_counter(self, action, reward):
        self.action_count[action] += 1
        self.action_reward[self.step, 0] = action   # 0,1,2
        self.action_reward[self.step, 1] = reward   # 0,1
        self.step += 1
        # self.q_star[action] += (reward - self.q_star[action]) / self.action_count[action]


    # 模拟运行
    def simulate(self, runs, steps):
        # 记录历史 reward，便于后面统计
        self.action_reward_list = []
        for run in trange(runs):
            # 每次run都独立，但是使用相同的参数
            self.reset(steps)
            # 测试 step 次
            for step in range(steps):
                action = self.select_action()
                reward = self.step_reward(action)
                self.update_counter(action, reward)
            # end for step
            self.action_reward_list.append(self.action_reward)
            #print(self.action_reward[0:100])
        

    # 统计值
    def summary(self):
        summary = []

        summary.append(np.arange(self.k_arms))
        summary.append(self.prob)

        # shape = (runs, step, [action:reward])
        action_reward_runs = np.array(self.action_reward_list)
        # 每个arm的选择次数
        action, count_per_action = np.unique(action_reward_runs[:,:,0], return_counts=True)
        summary.append(count_per_action)
        summary.append(summary[2]/summary[2].sum())

        # 每个 arm 上的总 reward
        rewards = []
        for action in range(self.k_arms):
            rewards.append(action_reward_runs[np.where(action_reward_runs[:,:,0]==action)].sum(axis=0)[1])
        summary.append(np.array(rewards))
        summary.append(summary[4]/summary[4].sum())
        summary.append(summary[4]/summary[2])

        summary2 = []
        mean_reward_step = action_reward_runs[:,:,1].mean(axis=0)
        summary2.append(mean_reward_step)

        best_action = np.argmax(self.prob)
        best_action_count_per_step = np.zeros(shape=(action_reward_runs.shape[0],action_reward_runs.shape[1]))
        for run in range(action_reward_runs.shape[0]):
            for step in range(action_reward_runs.shape[1]):
                if (action_reward_runs[run,step,0] == best_action):
                    best_action_count_per_step[run,step] += 1
        
        summary2.append(best_action_count_per_step.mean(axis=0))
        return summary, np.array(summary2), summary[4].sum()/summary[2].sum()

        '''
        np.set_printoptions(suppress=True)
        print(np.round(np.array(summary), 3))
        # 每次run的平均reward
        mean_reward_run = action_reward_runs[:,:,1].sum(axis=1).sum(axis=0) / action_reward_runs.shape[0]
        print(mean_reward_run)
        '''
