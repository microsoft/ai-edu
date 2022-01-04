import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_Optimistic_Initial(kab_base.KArmBandit):
    def __init__(self, k_arms=10, alpha=0.1, initial=0):
        super().__init__(k_arms=k_arms)
        self.alpha = alpha
        self.initial = initial

    def reset(self):
        super().reset()
        self.Q += self.initial

    def select_action(self):
        action = np.argmax(self.Q)
        return action
    
    def update_Q(self, action, reward):
        # 总次数(time)
        self.step += 1
        # 动作次数(action_count)
        self.action_count[action] += 1
        # 计算动作价值，固定步长更新
        self.Q[action] += self.alpha * (reward - self.Q[action])


        
if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Optimistic_Initial(k_arms, 0.1, 1))
    bandits.append(KAB_Optimistic_Initial(k_arms, 0.1, 3))
    bandits.append(KAB_Optimistic_Initial(k_arms, 0.1, 5))
    bandits.append(KAB_Optimistic_Initial(k_arms, 0.1, 7))

    labels = [
        'Initial(0.1,1), ',
        'Initial(0.1,3), ',
        'Initial(0.1,5), ',
        'Initial(0.1,7), ',
    ]
    title = 'Optimistic Initial'
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
