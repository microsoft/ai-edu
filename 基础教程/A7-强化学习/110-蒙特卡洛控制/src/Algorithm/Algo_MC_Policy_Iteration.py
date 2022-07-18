import numpy as np
import tqdm

# 策略迭代
class Policy_Iteration(object):
    def __init__(self, env, policy, episodes, gamma):
        self.policy = policy
        self.episodes = episodes
        self.env = env
        self.gamma = gamma

    # 初始化
    def initialize(self):
        print("初始化")
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.n
        self.Value = np.zeros((self.nS, self.nA))  # G 的总和
        self.Count = np.zeros((self.nS, self.nA))  # G 的数量

    # 策略评估
    def policy_evaluation(self, episodes):
        for _ in tqdm.trange(episodes):   # 多幕循环
            # 重置环境，开始新的一幕采样
            s = self.env.reset()
            Episode = []     # 一幕内的(状态,奖励)序列
            done = False
            while (done is False):            # 幕内循环
                action = np.random.choice(self.nA, p=self.policy[s])
                next_s, reward, done, info = self.env.step(action)
                Episode.append((s, action, reward))
                s = next_s

            num_step = len(Episode)
            G = 0
            # 从后向前遍历计算 G 值
            for t in range(num_step-1, -1, -1):
                s, a, r = Episode[t]
                G = self.gamma * G + r
                self.Value[s,a] += G     # 值累加
                self.Count[s,a] += 1     # 数量加 1

        self.Count[self.Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
        Q = self.Value / self.Count   # 求均值
        return Q

    # 策略改进(算法重写此函数)
    def policy_improvement(self, Q):
        pass

    # 策略迭代
    def policy_iteration(self):
        while True:
            self.initialize()
            print("策略评估")
            Q = self.policy_evaluation(self.episodes)
            print("策略改进")
            old_policy = self.policy.copy()
            self.policy_improvement(Q)
            if (old_policy == self.policy).all():
                print("新旧策略相等")
                break
        print("精准的策略评估")
        Q = self.policy_evaluation(20000)
        return Q, self.policy


class Policy_Iteration2(object):
    def __init__(self, env, policy, episodes, gamma):
        self.policy = policy
        self.episodes = episodes
        self.env = env
        self.gamma = gamma

    # 初始化
    def initialize(self):
        print("初始化")
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.n
        self.Value = np.zeros((self.nS, self.nA))  # G 的总和
        self.Count = np.zeros((self.nS, self.nA))  # G 的数量

    # 策略评估
    def policy_evaluation(self, episodes):
        for _ in tqdm.trange(episodes):   # 多幕循环
            # 重置环境，开始新的一幕采样
            s = self.env.reset()
            Episode = []     # 一幕内的(状态,奖励)序列
            done = False
            while (done is False):            # 幕内循环
                action = np.random.choice(self.nA, p=self.policy[s])
                next_s, reward, done, info = self.env.step(action)
                Episode.append((s, action, reward))
                s = next_s

            num_step = len(Episode)
            G = 0
            # 从后向前遍历计算 G 值
            for t in range(num_step-1, -1, -1):
                s, a, r = Episode[t]
                G = self.gamma * G + r
                self.Value[s,a] += G     # 值累加
                self.Count[s,a] += 1     # 数量加 1

        self.Count[self.Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
        Q = self.Value / self.Count   # 求均值
        return Q

    # 策略改进(算法重写此函数)
    def policy_improvement(self, Q):
        pass

    # 策略迭代
    def policy_iteration(self):
        self.initialize()
        print("策略评估")
        while True:
            Q = self.policy_evaluation(self.episodes)
            print("策略改进")
            old_policy = self.policy.copy()
            self.policy_improvement(Q)
            if (old_policy == self.policy).all():
                print("新旧策略相等")
                break
        print("精准的策略评估")
        Q = self.policy_evaluation(20000)
        return Q, self.policy
