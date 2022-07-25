import numpy as np
import tqdm

# 策略迭代
class Policy_Iteration(object):
    # 初始化
    def __init__(self, env, policy_args, gamma, rough, final, check_policy_method=0):
        self.rough_episodes = rough                 # 粗预测的分幕循环次数
        self.final_episodes = final                 # 细预测的分幕循环次数
        self.env = env                              # 环境
        self.gamma = gamma                          # 折扣
        self.check_method = check_policy_method     # 迭代终止条件
        self.nA = self.env.action_space.n           # 动作空间
        self.n_iteration = 0                        # 分幕循环次数总和
        if env.spec.id == 'Blackjack-v1':
            # 0-21=22, 0-10=11, A/noA=2, Hit/Stick=2
            self.Value = np.zeros((22, 11, 2, self.nA))   # G 的总和
            self.Count = np.zeros((22, 11, 2, self.nA))   # G 的数量
        else:
            self.nS = self.env.observation_space.n  # 状态空间
            self.Value = np.zeros((self.nS, self.nA))
            self.Count = np.zeros((self.nS, self.nA))
        self.policy = self.init_policy(policy_args)

    def init_policy(self, policy_args):
        if self.env.spec.id == 'Blackjack-v1':
            self.policy = np.zeros_like(self.Value)
            self.policy[:,:,:] = policy_args


    # 策略评估
    def policy_evaluation(self, episodes):
        for _ in tqdm.trange(episodes):   # 多幕循环
            self.n_iteration += 1
            # 重置环境，开始新的一幕采样
            s = self.env.reset()
            Episode = []     # 一幕内的(状态,奖励)序列
            done = False
            while (done is False):            # 幕内循环
                action = np.random.choice(self.nA, p=self.policy[s])
                next_s, reward, done, _ = self.env.step(action)
                Episode.append((s, action, reward))
                s = next_s
            # 从后向前遍历计算 G 值
            G = 0
            for t in range(len(Episode)-1, -1, -1):
                s, a, r = Episode[t]
                G = self.gamma * G + r
                self.Value[s,a] += G     # 值累加
                self.Count[s,a] += 1     # 数量加 1
        # 多幕循环结束，计算 Q 值
        self.Count[self.Count==0] = 1 # 把分母为0的填成1，主要是针对终止状态Count为0
        Q = self.Value / self.Count   # 求均值
        return Q

    # 策略改进(算法重写此函数)
    def policy_improvement(self, Q):
        pass

    # 策略迭代
    def policy_iteration(self):
        while True:
            print("策略评估")
            Q = self.policy_evaluation(self.rough_episodes)
            print("策略改进")
            old_policy = self.policy.copy()
            self.policy_improvement(Q)
            if self.check_policy_diff(old_policy, self.policy):
                print("新旧策略相等")
                break
        print("精准的策略评估")
        Q = self.policy_evaluation(self.final_episodes)
        return Q, self.policy

    def check_policy_diff(self, old_policy, new_policy):
        if (self.check_method == 0):
            return (old_policy == new_policy).all()
        elif (self.check_method == 1):
            old_arg = np.argmax(old_policy, axis=1)
            new_arg = np.argmax(new_policy, axis=1)
            return (old_arg == new_arg).all()
