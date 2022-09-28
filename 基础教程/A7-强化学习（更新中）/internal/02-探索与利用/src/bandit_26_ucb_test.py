import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import bandit_20_base as kab_base

class KAB_UCB_test(kab_base.KArmBandit):
    def __init__(self, k_arms=10, c=2):
        super().__init__(k_arms=k_arms)
        self.UCB_param = c

    def select_action(self):
        ucb = self.UCB_param * np.sqrt(np.log(self.step + 1) / (self.action_count + 1e-2))
        estimation = self.Q + ucb
        action = np.argmax(estimation)
        return action, self.Q, ucb
        
    # 模拟运行
    def simulate(self, runs, steps):
        # 记录历史 reward，便于后面统计
        rewards = np.zeros(shape=(runs, steps))
        actions = np.zeros(shape=(runs, steps), dtype=int)
        values = np.zeros(shape=(runs, steps, 2, self.k_arms))

        for r in range(runs):
            # 每次run都独立，但是使用相同的参数
            self.reset()
            # 测试 time 次
            for s in range(steps):
                action, mu, ucb = self.select_action()
                actions[r, s] = action
                values[r, s, 0] = mu
                values[r, s, 1] = ucb
                reward = self.step_reward(action)
                rewards[r, s] = reward
                self.update_Q(action, reward)
            # end for t
        # end for r
        return rewards, actions, values



if __name__ == "__main__":
    runs = 1
    steps = 100
    k_arms = 5

    bandit = KAB_UCB_test(k_arms, c=2)
    rewards, actions, values = bandit.simulate(runs, steps)
    
    for step in range(steps):
        mu = values[0,step,0]
        ucb = values[0,step,1]
        action = actions[0,step]
        print("step=",step, end=',');
        print("Q=", np.around(mu,2), end=',')
        print("UCB=", np.around(ucb,2),end=',')
        print("Q+UCB=", np.around(mu+ucb,2),end=',')
        print("action=", action)
