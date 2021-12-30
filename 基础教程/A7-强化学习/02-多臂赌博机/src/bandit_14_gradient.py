from matplotlib.pyplot import grid
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import gradient
from bandit_11_best_action import *


class K_ArmBandit_Gradient(K_ArmBandit_BestAction):
    def __init__(self, k_arms:int=10, epsilon:float=0, step:float=0.1, has_base:bool=False, grad_base:float=0):
        super().__init__(k_arms, epsilon)
        self.step = step
        self.has_base = has_base
        self.action_prob = 0
        self.average_reward = 0
        self.grad_base = grad_base

    def reset(self):
        super().reset()
        self.q_base += self.grad_base

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

    all_rewards = []
    all_best = []
    
    bandits = []
    bandits.append(K_ArmBandit_Gradient(k_arms=10, epsilon=0, step=0.1, has_base=True, grad_base=4))
    bandits.append(K_ArmBandit_Gradient(k_arms=10, epsilon=0, step=0.1, has_base=False, grad_base=4))
    bandits.append(K_ArmBandit_Gradient(k_arms=10, epsilon=0, step=0.4, has_base=True, grad_base=4))
    bandits.append(K_ArmBandit_Gradient(k_arms=10, epsilon=0, step=0.4, has_base=False, grad_base=4))

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

    for i in range(4):
        plt.plot(best_arm_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.grid()
    plt.show()