from matplotlib.pyplot import grid
from numpy.core.fromnumeric import argmax
from bandit_best_action import *

class K_ArmBandit_Gradient(K_ArmBandit_BestAction):
    def __init__(self, k: int, epsilon: float, step: float, has_base: bool, grid_base: float):
        super().__init__(k, epsilon)
        self.step = step
        self.has_base = has_base
        self.action_prob = 0
        self.average_reward = 0
        self.q_base += grid_base

    def select_action(self):
        # 小于epsilon, 执行随机探索行动
        if (np.random.rand() < self.epsilon):
            return np.random.choice(self.action_idx)

        q_star_exp = np.exp(self.q_star)
        self.action_prob = q_star_exp / np.sum(q_star_exp)
        action = np.random.choice(self.action_idx, p=self.action_prob)
        return action
    
    def update_q(self, action, reward):
        # 总次数(time)
        self.time += 1
        # 动作次数(action_count)
        self.action_count[action] += 1

        # 计算动作价值，偏好函数
        one_hot = np.zeros(self.k_arms)
        one_hot[action] = 1

        self.average_reward += (reward - self.average_reward) / self.time
        if (self.has_base):
            base = self.average_reward
        else:
            base = 0
        self.q_star += self.step * (reward - base) * (one_hot - self.action_prob)




if __name__ == "__main__":
    runs = 2000
    time = 1000
    k_arms = 10


    rewards = np.zeros((4, runs, time))
    best_arm = np.zeros(rewards.shape)
    
    i = 0
    # run 2000 次后，取平均
    for r in trange(runs):
        kab = K_ArmBandit_Gradient(k_arms, 0, 0.1, True, 4)
        # 测试 time 次
        for t in range(time):
            action = kab.select_action()
            reward = kab.step_reward(action)
            rewards[i, r, t] = reward
            if (action == kab.best_arm):
                best_arm[i, r, t] = 1    
    
    i = 1
    for r in trange(runs):
        kab = K_ArmBandit_Gradient(k_arms, 0, 0.1, False, 4)
        # 测试 time 次
        for t in range(time):
            action = kab.select_action()
            reward = kab.step_reward(action)
            rewards[i, r, t] = reward
            if (action == kab.best_arm):
                best_arm[i, r, t] = 1    
    
    i = 2
    for r in trange(runs):
        kab = K_ArmBandit_Gradient(k_arms, 0, 0.4, True, 4)
        # 测试 time 次
        for t in range(time):
            action = kab.select_action()
            reward = kab.step_reward(action)
            rewards[i, r, t] = reward
            if (action == kab.best_arm):
                best_arm[i, r, t] = 1    
    
    i = 3
    for r in trange(runs):
        kab = K_ArmBandit_Gradient(k_arms, 0, 0.4, False, 4)
        # 测试 time 次
        for t in range(time):
            action = kab.select_action()
            reward = kab.step_reward(action)
            rewards[i, r, t] = reward
            if (action == kab.best_arm):
                best_arm[i, r, t] = 1    


    best_arm_counts = best_arm.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)

    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']

    for i in range(4):
        plt.plot(best_arm_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.grid()
    plt.show()