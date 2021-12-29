import numpy as np

from bandit_01_best_action import *

class K_ArmBandit_Thompson(K_ArmBandit_BestAction):
    def __init__(self, k_arms, epsilon):
        super().__init__(k_arms, epsilon)


    def select_action(self):
        # 小于epsilon, 执行随机探索行动
        if (np.random.rand() < self.epsilon):
            # 从 k 个 arm 里随机选一个
            return np.random.choice(self.action_idx)

        a = np.random.beta(self.action_count+1, 1000-self.action_count)
        action = np.argmax(a)
        return action
        
    def update_q(self, action, reward):
        # 总次数(time)
        self.time += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        # 计算动作价值，采样平均
        self.q_star[action] += (reward - self.q_star[action]) / self.action_count[action]

if __name__ == "__main__":
    runs = 200
    time = 1000

    all_rewards = []
    all_best = []
    
    bandits = []
    bandits.append(K_ArmBandit_Thompson(k_arms=10, epsilon=0))

    for bandit in bandits:
        rewards, best_arm = bandit.simulate(runs, time)
        all_rewards.append(rewards)
        all_best.append(best_arm)

    best_arm_counts = np.array(all_best).mean(axis=1)
    mean_rewards = np.array(all_rewards).mean(axis=1)

    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_arm_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.grid()
    plt.show()