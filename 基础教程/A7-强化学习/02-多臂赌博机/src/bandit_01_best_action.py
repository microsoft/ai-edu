from bandit_00_base import *

class K_ArmBandit_BestAction(K_ArmBandit):
    def reset(self):
        super().reset()
        self.best_arm = np.argmax(self.q_base)

    def simulate(self, runs, time):
        # 记录历史 reward，便于后面统计
        rewards = np.zeros(shape=(runs, time))

        # 记录是否选到了均值最高的 arm
        best_arm = np.zeros(rewards.shape)

        for r in trange(runs):
            # 每次run都独立，但是使用相同的参数
            self.reset()
            # 测试 time 次
            for t in range(time):
                action = self.select_action()
                reward = self.step_reward(action)
                self.update_q(action, reward)
                rewards[r, t] = reward
                if (action == self.best_arm):
                    best_arm[r, t] = 1

            # end for t
        return rewards, best_arm


if __name__ == "__main__":

    epsilons = [0.0, 0.01, 0.05, 0.1]
    runs = 2000
    time = 1000
    k_arms = 10

    all_rewards = []
    all_best = []

    for eps in epsilons:
        bandit = K_ArmBandit_BestAction(k_arms, eps)
        rewards, best_arm = bandit.simulate(runs, time)
        all_rewards.append(rewards)
        all_best.append(best_arm)


    # 取2000次的平均
    best_arm_counts = np.array(all_best).mean(axis=1)
    mean_rewards = np.array(all_rewards).mean(axis=1)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, mean_rewards in zip(epsilons, mean_rewards):
        plt.plot(mean_rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_arm_counts):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.grid()

    plt.show()
