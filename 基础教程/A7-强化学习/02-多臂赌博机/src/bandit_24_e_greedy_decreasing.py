import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_E_Greedy_Descreasing(kab_base.KArmBandit):
    def __init__(self, k_arms=10, epsilon_scope=[0.1,0.0001], steps=1000):
        super().__init__(k_arms=k_arms)
        self.epsilon_range = np.linspace(epsilon_scope[0], epsilon_scope[1], num=steps)

    def select_action(self):
        if (np.random.rand() < self.epsilon_range[self.step]):
            action = np.random.randint(self.k_arms)
        else:
            action = np.argmax(self.Q)
        return action
    
        
if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    all_rewards = []
    all_best = []
    all_actions = []

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_E_Greedy_Descreasing(k_arms, [0.1,0.0001], steps))
    bandits.append(KAB_E_Greedy_Descreasing(k_arms, [0.1,0.001], steps))
    bandits.append(KAB_E_Greedy_Descreasing(k_arms, [0.1,0.005], steps))
    bandits.append(KAB_E_Greedy_Descreasing(k_arms, [0.1,0.01], steps))

    labels = [
        'E-Greedy-D(0.0001), ',
        'E-Greedy-D(0.001), ',
        'E-Greedy-D(0.005), ',
        'E-Greedy-D(0.01), ',
    ]

    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels)
