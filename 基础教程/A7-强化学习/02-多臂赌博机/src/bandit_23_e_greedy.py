import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_E_Greedy(kab_base.KArmBandit):
    def __init__(self, k_arms=10, epsilon=0.1):
        super().__init__(k_arms=k_arms)
        self.epsilon = epsilon

    def select_action(self):
        if (np.random.rand() < self.epsilon):
            action = np.random.randint(self.k_arms)
        else:
            action = np.argmax(self.Q)
        return action
    
        
if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_E_Greedy(k_arms, 0.01))
    bandits.append(KAB_E_Greedy(k_arms, 0.02))
    bandits.append(KAB_E_Greedy(k_arms, 0.05))
    bandits.append(KAB_E_Greedy(k_arms, 0.1))

    labels = [
        'E-Greedy(0.01), ',
        'E-Greedy(0.02), ',
        'E-Greedy(0.05), ',
        'E-Greedy(0.1), ',
    ]
    title = "E-Greedy"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
