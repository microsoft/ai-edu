import numpy as np
import bandit_20_base as kab_base

class KAB_Softmax_copy(kab_base.KArmBandit):
    def __init__(self, k_arms=10, alpha:float=0.1):
        super().__init__(k_arms=k_arms)
        self.alpha = alpha

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
        self.average_reward += (reward - self.average_reward) / self.steps
        # 计算动作价值
        self.Q[action] += self.alpha * (reward - self.average_reward) * self.P[action]

        '''
        for i in range(self.k_arms):
            if (i != action):
                self.Q[i] += self.alpha * (-self.average_reward) * self.P[i]
        '''
        #self.Q[action] += self.alpha * (reward - self.Q.mean()) * (1-self.P[action])
        # Q(a) = Q(a) + alpha * (R - Q.mean()) * P(a)

if __name__ == "__main__":
    runs = 2000
    steps = 1000
    k_arms = 10 
    
    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Softmax_copy(k_arms, alpha=0.6))
    bandits.append(KAB_Softmax_copy(k_arms, alpha=0.7))
    bandits.append(KAB_Softmax_copy(k_arms, alpha=0.8))
    bandits.append(KAB_Softmax_copy(k_arms, alpha=0.9))

    labels = [
        'Softmax(0.6)',
        'Softmax(0.7)',
        'Softmax(0.8)',
        'Softmax(0.9)',
    ]

    title = 'Softmax'
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
