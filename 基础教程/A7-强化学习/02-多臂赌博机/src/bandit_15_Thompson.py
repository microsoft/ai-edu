import numpy as np

from bandit_11_best_action import *

class K_ArmBandit_Thompson(K_ArmBandit_BestAction):
    def reset(self):
        super().reset()

        self.a = np.ones(self.action_count.shape)
        self.b = np.ones(self.action_count.shape)

    def select_action(self):
        # 小于epsilon, 执行随机探索行动
        if (np.random.rand() < self.epsilon):
            # 从 k 个 arm 里随机选一个
            return np.random.choice(self.action_idx)

        beta = np.random.beta(self.a, self.b)
        action = np.argmax(beta)
        return action
        
    def update_q(self, action, reward):
        # 总次数(time)
        self.time += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        #if (reward > self.q_base[self.best_arm]):
        if (reward > self.q_base[self.best_arm]):
            self.a[action] += 1
        else:
            self.b[action] += 1
        # 计算动作价值，采样平均
        self.q_star[action] += (reward - self.q_star[action]) / self.action_count[action]

if __name__ == "__main__":
    runs = 1000
    time = 1000

    all_rewards = []
    all_best = []
    
    bandits = []
    bandits.append(K_ArmBandit_BestAction(k_arms=10, epsilon=0.1))
    bandits.append(K_ArmBandit_Thompson(k_arms=10, epsilon=0))


    for bandit in bandits:
        rewards, best_arm = bandit.simulate(runs, time)
        all_rewards.append(rewards)
        all_best.append(best_arm)

    best_arm_counts = np.array(all_best).mean(axis=1)
    mean_rewards = np.array(all_rewards).mean(axis=1)

    labels = [r'$\epsilon = 0.1$',
            r'$\epsilon = 0$, Thompson']


    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for i in range(len(bandits)):
        plt.plot(mean_rewards[i], label=labels[i])
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for i in range(len(bandits)):
        plt.plot(best_arm_counts[i], label=labels[i])
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.grid()
    plt.show()
