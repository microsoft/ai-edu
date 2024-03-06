import numpy as np
import bandit_3_Base as kab_base

class KAB_E_Greedy(kab_base.KArmBandit):
    def __init__(self, k_arms=10, epsilon=0.1):
        super().__init__(k_arms=k_arms)
        self.epsilon = epsilon  # 非贪心概率

    def select_action(self):
        if (np.random.random_sample() < self.epsilon):
            action = np.random.randint(self.k_arms) # 随机选择动作进行探索
        else:
            action = np.argmax(self.Q)  # 贪心选择目前最好的动作进行利用
        return action
    
        
if __name__ == "__main__":
    runs = 2000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_E_Greedy(k_arms, epsilon=0.01))
    bandits.append(KAB_E_Greedy(k_arms, epsilon=0.05))
    bandits.append(KAB_E_Greedy(k_arms, epsilon=0.10))
    bandits.append(KAB_E_Greedy(k_arms, epsilon=0.20))

    labels = [
        'E-Greedy(0.01)',
        'E-Greedy(0.05)',
        'E-Greedy(0.10)',
        'E-Greedy(0.20)',
    ]
    title = "E-Greedy"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
