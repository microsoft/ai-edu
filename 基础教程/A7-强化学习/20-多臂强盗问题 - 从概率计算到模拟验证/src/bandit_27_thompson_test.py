import matplotlib.pyplot as plt
import numpy as np
from bandit_27_Thompson import KAB_Thompson
import scipy.stats as ss


class KAB_Thompson_test(KAB_Thompson):
    def __init__(self, k_arms=10, method=0):
        super().__init__(k_arms=k_arms, method=method)

    def select_action(self):
        beta = np.random.beta(self.alpha, self.beta)
        action = np.argmax(beta)
        return action, beta

    def simulate(self, runs, steps):
        rewards = np.zeros(shape=(runs, steps))
        actions = np.zeros(shape=(runs, steps), dtype=int)
        win_loss = np.zeros(shape=(runs, steps, 2, self.k_arms))
        beta_samples = np.zeros(shape=(runs, steps, self.k_arms))

        for r in range(runs):
            # 每次run都清零计算 q 用的统计数据，并重新初始化奖励均值
            self.reset()
            self.alpha += 1
            self.beta += 1
            # 测试 time 次
            for s in range(steps):
                win_loss[r, s, 0] = self.alpha
                win_loss[r, s, 1] = self.beta
                action, beta = self.select_action()
                actions[r, s] = action
                beta_samples[r, s] = beta
                reward = self.pull_arm(action)
                rewards[r, s] = reward
                self.update_Q(action, reward)
        return rewards, actions, win_loss, beta_samples

if __name__ == "__main__":
    runs = 1
    steps = 20
    k_arms = 3

    np.random.seed(5)
    bandit = KAB_Thompson_test(k_arms, 0)
    rewards, actions, win_loss, beta = bandit.simulate(runs, steps)

    grid = plt.GridSpec(nrows=3, ncols=5)
    x = np.linspace(0,1,num=101)
    for i in range(steps):
        s = str.format("step:{5}\twin:loss={0}:{1}\tbeta={2}\ta={3}\tr={4}", 
            np.round(win_loss[0,i,0],2),
            np.round(win_loss[0,i,1],2),
            np.round(beta[0,i], 2),
            actions[0,i],
            np.round(rewards[0,i],2), i)
        print(s)
        if (i >= 15):
            continue
        plt.subplot(grid[int(i/5), i%5])
        for j in range(k_arms):
            beta_pdf = ss.beta.pdf(x, win_loss[0,i,0,j], win_loss[0,i,1,j])
            plt.plot(x, beta_pdf, label=str(j))
            plt.scatter(beta[0,i,j], 0)
            plt.title(str.format("step:{0},a={1},r={2:.2f}",i,actions[0,i],rewards[0,i]))
            plt.grid()
            plt.legend()
    plt.show()
