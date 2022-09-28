import numpy as np
import math
import bandit_23_Base as kab_base

class KAB_UCB(kab_base.KArmBandit):
    def __init__(self, k_arms=10, c=1):
        super().__init__(k_arms=k_arms)
        self.C = c

    def select_action(self):
        ucb = self.C * np.sqrt(math.log(self.steps + 1) / (self.action_count + 1e-2))
        estimation = self.Q + ucb
        action = np.argmax(estimation)
        return action
        
if __name__ == "__main__":
    runs = 2000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_UCB(k_arms, c=0.5))
    bandits.append(KAB_UCB(k_arms, c=0.7))
    bandits.append(KAB_UCB(k_arms, c=1))
    bandits.append(KAB_UCB(k_arms, c=1.2))

    labels = [
        'UCB(c=0.5)',
        'UCB(c=0.7)',
        'UCB(c=1.0)',
        'UCB(c=1.2)'
    ]
    title = "UCB"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
