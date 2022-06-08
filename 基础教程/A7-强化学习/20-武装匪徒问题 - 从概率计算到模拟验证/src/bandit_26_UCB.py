import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_UCB(kab_base.KArmBandit):
    def __init__(self, k_arms=10, c=2):
        super().__init__(k_arms=k_arms)
        self.UCB = c

    def select_action(self):
        ucb = self.UCB * np.sqrt(np.log(self.step + 1) / (self.action_count + 1e-2))
        estimation = self.Q + ucb
        action = np.argmax(estimation)
        return action
        
if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_UCB(k_arms, c=0.1))
    bandits.append(KAB_UCB(k_arms, c=0.5))
    bandits.append(KAB_UCB(k_arms, c=1))
    bandits.append(KAB_UCB(k_arms, c=2))

    labels = [
        'UCB(c=0.1), ',
        'UCB(c=0.5), ',
        'UCB(c=1), ',
        'UCB(c=2), ',
    ]
    title = "UCB"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
