import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_Thompson(kab_base.KArmBandit):
    def __init__(self, k_arms=10, threshold=0):
        super().__init__(k_arms=k_arms)
        self.threshold = threshold

    def reset(self):
        super().reset()
        self.win = np.ones(self.action_count.shape)
        self.loss = np.ones(self.action_count.shape)


    def select_action(self):
        beta = np.random.beta(self.win, self.loss)
        action = np.argmax(beta)
        return action

    def update_Q(self, action, reward):
        super().update_Q(action, reward)
        if (self.threshold == -1):
            if (reward > self.q_base[self.k_arms-1]):
                self.win[action] += 1
            else:
                self.loss[action] += 1
        else:  # self.threshold > 0
            if (reward > self.threshold):
                self.win[action] += 1
            else:
                self.loss[action] += 1

if __name__ == "__main__":
    runs = 1000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Thompson(k_arms, -1))
    bandits.append(KAB_Thompson(k_arms, 0.1))
    bandits.append(KAB_Thompson(k_arms, 0.5))
    bandits.append(KAB_Thompson(k_arms, 1))

    labels = [
        'KAB_Thompson(-1), ',
        'KAB_Thompson(0.1), ',
        'KAB_Thompson(0.5), ',
        'KAB_Thompson(1), ',
    ]
    title = "Thompson"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
