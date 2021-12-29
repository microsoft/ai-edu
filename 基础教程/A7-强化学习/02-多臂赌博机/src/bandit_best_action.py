from bandit import *

class K_ArmBandit_BestAction(K_ArmBandit):
    def __init__(self, k, epsilon):
        super().__init__(k, epsilon)
        self.best_arm = np.argmax(self.q_base)

if __name__ == "__main__":

    epsilons = [0.0, 0.01, 0.05, 0.1]
    runs = 200
    time = 1000
    k_arms = 10

    total_action_count = np.zeros(k_arms)
    total_q_estimation = np.zeros(k_arms)

    rewards = np.zeros((len(epsilons), runs, time))
    best_arm = np.zeros(rewards.shape)

    for i, eps in enumerate(epsilons):
        # kab = K_ArmBandit(k_arms, eps)
        # run 2000 次后，取平均
        for r in trange(runs):
            kab = K_ArmBandit_BestAction(k_arms, eps)
            # 测试 time 次
            for t in range(time):
                action = kab.select_action()
                reward = kab.step_reward(action)
                rewards[i, r, t] = reward
                if (action == kab.best_arm):
                    best_arm[i, r, t] = 1
            # end for t
            total_action_count += kab.action_count
            total_q_estimation += kab.q_star

    print(total_action_count)
    print(total_q_estimation)

    # 取2000次的平均
    best_arm_counts = best_arm.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)

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

