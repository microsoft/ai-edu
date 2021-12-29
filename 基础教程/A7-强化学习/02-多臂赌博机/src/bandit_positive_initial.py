from bandit_best_action import *

class K_ArmBandit_PostiveInitial(K_ArmBandit_BestAction):
    def __init__(self, k, epsilon, initial, step):
        super().__init__(k, epsilon)
        # 乐观初始值，即令初始值为正
        self.q_star = np.zeros(self.k_arms) + initial
        self.step = step

    def update_q(self, action, reward):
        # 总次数(time)
        self.time += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        # 计算价值
        self.q_star[action] = self.q_star[action] + self.step * (reward - self.q_star[action])

if __name__ == "__main__":

    runs = 2000
    time = 1000
    k_arms = 10

    total_action_count = np.zeros(k_arms)
    total_q_estimation = np.zeros(k_arms)

    rewards = np.zeros((2, runs, time))
    best_arm = np.zeros(rewards.shape)

    eps = 0.1
    initial = 0
    step = 0.1
    i = 0
    # run 2000 次后，取平均
    for r in trange(runs):
        kab = K_ArmBandit_PostiveInitial(k_arms, eps, initial, step)
        # 测试 time 次
        for t in range(time):
            action = kab.select_action()
            reward = kab.step_reward(action)
            rewards[i, r, t] = reward
            if (action == kab.best_arm):
                best_arm[i, r, t] = 1
        # end for t

    eps = 0
    initial = 5
    step = 0.1
    i = 1
    # run 2000 次后，取平均
    for r in trange(runs):
        kab = K_ArmBandit_PostiveInitial(k_arms, eps, initial, step)
        # 测试 time 次
        for t in range(time):
            action = kab.select_action()
            reward = kab.step_reward(action)
            rewards[i, r, t] = reward
            if (action == kab.best_arm):
                best_arm[i, r, t] = 1
        # end for t


    # 取2000次的平均
    best_arm_counts = best_arm.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)

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

