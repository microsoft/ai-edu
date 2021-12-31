import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_Softmax(kab_base.KArmBandit):
    def __init__(self, k_arms=10, alpha:float=0.1, has_base:bool=False, grad_base:float=0):
        super().__init__(k_arms=k_arms)
        self.alpha = alpha
        self.has_base = has_base
        self.action_prob = 0
        self.grad_base = grad_base


    def reset(self):
        super().reset()
        self.q_base += self.grad_base
        self.average_reward = 0


    def select_action(self):
        q_exp = np.exp(self.Q - np.max(self.Q))
        self.action_prob = q_exp / np.sum(q_exp)
        action = np.random.choice(self.k_arms, p=self.action_prob)
        return action
  

    def update_Q(self, action, reward):
        # 总次数(time)
        self.step += 1
        # 动作次数(action_count)
        self.action_count[action] += 1


        # 计算动作价值，偏好函数
        one_hot = np.zeros(self.k_arms)
        one_hot[action] = 1
        self.average_reward += (reward - self.average_reward) / self.step
        if (self.has_base):
            base = self.average_reward
        else:
            base = 0
        self.Q += self.alpha * (reward - base) * (one_hot - self.action_prob)

        
if __name__ == "__main__":
    runs = 2000
    steps = 1000
    k_arms = 10

    all_rewards = []
    all_best = []
    all_actions = []

    bandits:kab_base.KArmBandit = []
    bandits.append(KAB_Softmax(k_arms, alpha=0.1, has_base=True, grad_base=0))
    bandits.append(KAB_Softmax(k_arms, alpha=0.1, has_base=False, grad_base=0))
    bandits.append(KAB_Softmax(k_arms, alpha=0.4, has_base=True, grad_base=0))
    bandits.append(KAB_Softmax(k_arms, alpha=0.4, has_base=False, grad_base=0))

    labels = [
        'E-Greedy-D(0.1,True,0), ',
        'E-Greedy-D(0.1,False,0), ',
        'E-Greedy-D(0.4,True,0), ',
        'E-Greedy-D(0.4,False,0), ',
    ]

    kab_base.mp_simulate(bandits, k_arms, runs, steps, labels)
