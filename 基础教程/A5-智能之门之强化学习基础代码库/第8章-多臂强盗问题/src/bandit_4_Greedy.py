import numpy as np
import bandit_3_Base as kab_base

class KAB_Greedy(kab_base.KArmBandit):
    def __init__(self, k_arms=10, try_steps=10):
        super().__init__(k_arms=k_arms)
        self.try_steps = try_steps  # 试探次数

    def select_action(self):
        if (self.steps < self.try_steps):
            action = np.random.randint(self.k_arms) # 随机选择动作
        else:
            action = np.argmax(self.Q)  # 贪心选择目前最好的动作
        return action


if __name__ == "__main__":
    runs = 200
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Greedy(k_arms, try_steps=10))
    bandits.append(KAB_Greedy(k_arms, try_steps=20))
    bandits.append(KAB_Greedy(k_arms, try_steps=40))
    bandits.append(KAB_Greedy(k_arms, try_steps=80))

    labels = [
        'Greedy(10)',
        'Greedy(20)',
        'Greedy(40)',
        'Greedy(80)'
    ]
    title = "Greedy"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
