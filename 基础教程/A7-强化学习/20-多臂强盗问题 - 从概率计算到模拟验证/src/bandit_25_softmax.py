import numpy as np
import bandit_20_base as kab_base

class KAB_Softmax(kab_base.KArmBandit):
    def __init__(self, k_arms=10, alpha:float=0.1):
        super().__init__(k_arms=k_arms)
        self.alpha = alpha
        self.P = 0

    def reset(self):
        super().reset()
        self.average_reward = 0

    def select_action(self):
        q_exp = np.exp(self.Q - np.max(self.Q))     # 所有的值都减去最大值
        self.P = q_exp / np.sum(q_exp)    # softmax 实现
        action = np.random.choice(self.k_arms, p=self.P)  # 按概率选择动作
        return action

    def update_Q(self, action, reward):
        self.steps += 1 # 迭代次数
        self.action_count[action] += 1  # 动作次数(action_count)
        # 计算动作价值
        one_hot = np.zeros(self.k_arms)
        one_hot[action] = 1
        self.average_reward += (reward - self.average_reward) / self.steps
        self.Q += self.alpha * (reward - self.average_reward) * (one_hot - self.P)

    # def simulate(self, runs, steps):
    #     rewards, best_action, actions = super().simulate(runs, steps)
    #     return rewards, best_action, actions

if __name__ == "__main__":
    runs = 2000
    steps = 1000
    k_arms = 10 
    
    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Softmax(k_arms, alpha=0.10))
    bandits.append(KAB_Softmax(k_arms, alpha=0.15))
    bandits.append(KAB_Softmax(k_arms, alpha=0.20))
    bandits.append(KAB_Softmax(k_arms, alpha=0.25))

    labels = [
        'Softmax(0.10)',
        'Softmax(0.15)',
        'Softmax(0.20)',
        'Softmax(0.25)',
    ]

    title = 'Softmax'
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
