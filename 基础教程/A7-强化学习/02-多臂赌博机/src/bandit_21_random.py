import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_Random(kab_base.KArmBandit):
    def select_action(self):
        action = np.random.randint(self.k_arms)
        return action
    
        
if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    all_rewards = []
    all_best = []
    all_actions = []

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Random(k_arms))
    bandits.append(KAB_Random(k_arms))
    bandits.append(KAB_Random(k_arms))
    bandits.append(KAB_Random(k_arms))

    labels = [
        'random ',
        'random ',
        'random ',
        'random ',
    ]
    title = "Random"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
