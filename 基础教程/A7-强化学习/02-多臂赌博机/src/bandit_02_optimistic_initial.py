from bandit_01_best_action import *

class K_ArmBandit_OptimisticInitial(K_ArmBandit_BestAction):
    def __init__(self, k_arms=10, epsilon=0, initial=0, step=0):
        super().__init__(k_arms, epsilon)
        self.initial = initial # 乐观初始值，即令初始值为正
        self.step = step

    def reset(self):
        super().reset()
        self.q_star = np.zeros(self.k_arms) + self.initial
        self.best_arm = np.argmax(self.q_base)

    def update_q(self, action, reward):
        # 总次数(time)
        self.time += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        # 计算价值
        self.q_star[action] += self.step * (reward - self.q_star[action])

if __name__ == "__main__":

    runs = 2000
    time = 1000
    k_arms = 10

    all_rewards = []
    all_best = []

    bandits = []
    bandits.append(K_ArmBandit_OptimisticInitial(k_arms=10, epsilon=0.1, initial=0, step=0.1))
    bandits.append(K_ArmBandit_OptimisticInitial(k_arms=10, epsilon=0.0, initial=5, step=0.1))

    for bandit in bandits:
        rewards, best_arm = bandit.simulate(runs, time)
        all_rewards.append(rewards)
        all_best.append(best_arm)

    # 取2000次的平均
    best_arm_counts = np.array(all_best).mean(axis=1)
    mean_rewards = np.array(all_rewards).mean(axis=1)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for mean_reward in mean_rewards:
        plt.plot(mean_reward)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for counts in best_arm_counts:
        plt.plot(counts)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.grid()
    plt.show()

