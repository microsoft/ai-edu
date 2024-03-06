import numpy as np
from bandit_6_UCB import KAB_UCB
import matplotlib.pyplot as plt


class KAB_UCB_test(KAB_UCB):
    def select_action(self):
        ucb = self.C * np.sqrt(np.log(self.steps + 1) / (self.action_count + 1e-2))
        estimation = self.Q + ucb
        action = np.argmax(estimation)
        return action, self.Q, ucb
        
    # 模拟运行
    def simulate(self, runs, steps):
        # 记录历史 reward，便于后面统计
        rewards = np.zeros(shape=(runs, steps))
        actions = np.zeros(shape=(runs, steps), dtype=int)
        values = np.zeros(shape=(runs, steps, 2, self.k_arms))

        for r in range(runs):
            # 每次run都独立，但是使用相同的参数
            self.reset()
            # 测试 time 次
            for s in range(steps):
                action, mu, ucb = self.select_action()
                actions[r, s] = action
                values[r, s, 0] = mu
                values[r, s, 1] = ucb
                reward = self.pull_arm(action)
                rewards[r, s] = reward
                self.update_Q(action, reward)
            # end for t
        # end for r
        return rewards, actions, values


if __name__ == "__main__":
    runs = 1
    steps = 100
    k_arms = 3

    np.random.seed(13)
    bandit = KAB_UCB_test(k_arms, c=1)
    rewards, actions, values = bandit.simulate(runs, steps)
    
    for step in range(steps):
        mu = values[0,step,0]
        ucb = values[0,step,1]
        action = actions[0,step]
        reward = rewards[0,step]

        s = str.format("step={0:2d}, Q={1}, UCB={2}, Q+UCB={3}, a={4}, r={5:.2f}", 
            step, np.around(mu,2), np.around(ucb,2), np.around(mu+ucb,2), action, reward)
        print(s)


    grid = plt.GridSpec(nrows=1, ncols=6)

    for step in range(steps):
        if step > 5:
            continue
        mu = values[0,step,0]
        ucb = values[0,step,1]
        action = actions[0,step]
        reward = rewards[0,step]

        plt.subplot(grid[0, step])
        plt.bar([0,1,2], mu + ucb, width=0.5, bottom=mu, hatch=['o','/','x'])
        plt.title(str.format("a={0},r={1:.2f}", action, reward))

    plt.show()
