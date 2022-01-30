import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_Thompson(kab_base.KArmBandit):
    def __init__(self, k_arms=10, threshold=0):
        super().__init__(k_arms=k_arms)
        self.threshold = threshold

    def reset(self):
        super().reset()
        self.a = np.ones(self.action_count.shape)
        self.b = np.ones(self.action_count.shape)

    def select_action(self):
        beta = np.random.beta(self.a, self.b)
        action = np.argmax(beta)
        return action

    def update_Q(self, action, reward):
        super().update_Q(action, reward)
        if (self.threshold == -1):
            if (reward > self.q_base[self.k_arms-1]):
                self.a[action] += 1
            else:
                self.b[action] += 1
        else:  # self.threshold > 0
            if (reward > self.threshold):
                self.a[action] += 1
            else:
                self.b[action] += 1

    def simulate(self, runs, steps):
        # 记录历史 reward，便于后面统计
        A = np.zeros(shape=(runs, steps, self.k_arms), dtype=int)
        B = np.zeros(shape=(runs, steps, self.k_arms), dtype=int)
        actions = np.zeros(shape=(runs, steps), dtype=int)
        for r in range(runs):
            # 每次run都独立，但是使用相同的参数
            self.reset()
            # 测试 time 次
            for s in range(steps):
                action = self.select_action()
                A[r, s] = self.a
                B[r, s] = self.b
                actions[r, s] = action
                reward = self.step_reward(action)
                self.update_Q(action, reward)
            # end for t
        # end for r
        return A, B, actions

if __name__ == "__main__":
    runs = 1
    steps = 1000
    k_arms = 5

    bandit = KAB_Thompson(k_arms, 0.5)
    A, B, actions = bandit.simulate(runs, steps)

    import scipy.stats as ss
    x = np.linspace(0,1,num=101)
    for i in range(100):
        if (i % 10 == 0):
            for j in range(k_arms):
                beta = ss.beta.pdf(x, A[0,i,j], B[0,i,j])
                plt.plot(x, beta, label=str(j))
                plt.title(str.format("step:{0}, action:{1}, {2}:{3}",i,actions[0,i], A[0,i,j], B[0,i,j]))
            plt.grid()
            plt.legend()
            plt.show()

