import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_Softmax(kab_base.KArmBandit):
    def __init__(self, k_arms=10, alpha:float=0.1, temperature=1):
        super().__init__(k_arms=k_arms)
        self.alpha = alpha
        self.action_prob = 0
        self.tmpt = temperature


    def reset(self):
        super().reset()
        self.average_reward = 0


    def select_action(self):
        q_exp = np.exp((self.Q - np.max(self.Q))/self.tmpt)
        self.action_prob = q_exp / np.sum(q_exp)
        action = np.random.choice(self.k_arms, p=self.action_prob)
        return action
  
    
    def update_Q(self, action, reward):
        # 总次数(time)
        self.step += 1
        # 动作次数(action_count)
        self.action_count[action] += 1

        # 计算动作价值，偏好函数
        one_hot = np.zeros(self.k_arms)
        one_hot[action] = 1
        self.average_reward += (reward - self.average_reward) / self.step
        self.Q += self.alpha * (reward - self.average_reward) * (one_hot - self.action_prob)
    
        
    def simulate(self, runs, steps):
        rewards, best_action, actions = super().simulate(runs, steps)
        return rewards, best_action, actions

if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10 
    
    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Softmax(k_arms, alpha=0.1, temperature=1))
    bandits.append(KAB_Softmax(k_arms, alpha=0.2, temperature=1))
    bandits.append(KAB_Softmax(k_arms, alpha=0.3, temperature=1))
    bandits.append(KAB_Softmax(k_arms, alpha=0.4, temperature=1))

    labels = [
        'Softmax(0.1,1), ',
        'Softmax(0.2,1), ',
        'Softmax(0.3,1), ',
        'Softmax(0.4,1), ',
    ]

    title = r'Softmax ($\alpha$=0.1,0.2,0.3,0.4)'
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)

    
    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Softmax(k_arms, alpha=0.15, temperature=1))
    bandits.append(KAB_Softmax(k_arms, alpha=0.2, temperature=1))
    bandits.append(KAB_Softmax(k_arms, alpha=0.25, temperature=1))
    bandits.append(KAB_Softmax(k_arms, alpha=0.3, temperature=1))

    labels = [
        'Softmax(0.15,1), ',
        'Softmax(0.2,1), ',
        'Softmax(0.25,1), ',
        'Softmax(0.3,1), ',
    ]
    title = r'Softmax ($\alpha$=0.15,0.2,0.25,0.3)'
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
    
    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Softmax(k_arms, alpha=0.15, temperature=0.5))
    bandits.append(KAB_Softmax(k_arms, alpha=0.15, temperature=0.75))
    bandits.append(KAB_Softmax(k_arms, alpha=0.15, temperature=1))
    bandits.append(KAB_Softmax(k_arms, alpha=0.15, temperature=2))

    labels = [
        'Softmax(0.15,0.6), ',
        'Softmax(0.15,0.8), ',
        'Softmax(0.15,1), ',
        'Softmax(0.15,2), ',
    ]
    title = r'Softmax ($\tau$=0.6,0.8,1,2)'
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
    
