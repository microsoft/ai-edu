import numpy as np
import bandit_3_Base as kab_base


class KAB_Thompson(kab_base.KArmBandit):
    def __init__(self, k_arms=10, method=0):
        super().__init__(k_arms=k_arms)
        self.method = method    # -1: 与自身均值比；0：与所有均值比；>0：期望的门限值

    def reset(self):
        super().reset()
        self.total_average = 0
        self.alpha = np.ones(self.k_arms)
        self.beta = np.ones(self.k_arms)

    def select_action(self):
        p_beta = np.random.beta(self.alpha, self.beta)
        action = np.argmax(p_beta)
        return action

    def update_Q(self, action, reward):
        super().update_Q(action, reward)
        self.total_average += (reward - self.total_average) / self.steps

        is_win = False        
        if (self.method == -1): # 与整体均值比较
            if (reward >= self.total_average):
                is_win = True
        else:   # 与输入的期望值比较
            if (reward >= self.method):
                is_win = True
        # 用reward计数
        if is_win:
            self.alpha[action] += abs(reward)
        else:
            self.beta[action] += abs(reward)


if __name__ == "__main__":
    runs = 2000
    steps = 1000
    k_arms = 10

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Thompson(k_arms, -1))
    bandits.append(KAB_Thompson(k_arms, 0))
    bandits.append(KAB_Thompson(k_arms, 0.5))
    bandits.append(KAB_Thompson(k_arms, 0.8))

    labels = [
        'KAB_Thompson(-1)',
        'KAB_Thompson(0.0)',
        'KAB_Thompson(0.5)',
        'KAB_Thompson(0.8)',
    ]
    title = "Thompson"
    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels, title)
