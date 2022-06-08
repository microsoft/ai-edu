import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base
import bandit_21_random as kab_random

class KAB_Greedy(kab_base.KArmBandit):
    def __init__(self, k_arms=10, try_steps=10):
        super().__init__(k_arms=k_arms)
        self.try_steps = try_steps

    def select_action(self):
        if (self.step < self.try_steps):
            action = np.random.randint(self.k_arms)
        else:
            action = np.argmax(self.Q)
        return action
    
        
if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, 10))
    bandits.append(KAB_Greedy(k_arms, 15))
    bandits.append(KAB_Greedy(k_arms, 20))
    bandits.append(KAB_Greedy(k_arms, 25))

    labels = [
        'Greedy(10)',
        'Greedy(15)',
        'Greedy(20)',
        'Greedy(25)'
    ]
    title = "Greedy"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)






