from bandit_best_action import *

class bandit_UCB(K_ArmBandit_BestAction):
    def __init__(self, k, epsilon, ucb_param):
        super().__init__(k, epsilon)
        self.UCB = ucb_param

    def select_action(self):
        # 小于epsilon, 执行随机探索行动
        if (np.random.rand() < self.epsilon):
            return np.random.choice(self.action_idx)

        estimation = self.q_star + self.UCB * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))

        # 获得目前可以估计的最大值
        q_best_value = np.max(estimation)
        # 可能同时有多个最大值，获得它们在数组中的索引位置
        best_actions = np.where(q_best_value == estimation)[0]
        # 从多个最大值（索引）中随机选择一个作为下一步的动作
        action = np.random.choice(best_actions)
        return action

if __name__ == "__main__":

    runs = 200
    time = 1000
    k_arms = 10

    total_action_count = np.zeros(k_arms)
    total_q_estimation = np.zeros(k_arms)

    rewards = np.zeros((2, runs, time))
    best_arm = np.zeros(rewards.shape)

    eps = 0.1
    i = 0
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

    eps = 0
    UCB_param = 2
    i = 1
    # run 2000 次后，取平均
    for r in trange(runs):
        kab = bandit_UCB(k_arms, eps, UCB_param)
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

