from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from bandit_5_Softmax import KAB_Softmax
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False

class KAB_Softmax_test(KAB_Softmax):
    def __init__(self, k_arms=10, alpha:float=0.1):
        super().__init__(k_arms=k_arms, alpha=alpha)
        self.Ps = []
        self.Qs = []

    def select_action(self):
        q_exp = np.exp(self.Q - np.max(self.Q))     # 所有的值都减去最大值
        self.P = q_exp / np.sum(q_exp)    # softmax 实现
        action = np.random.choice(self.k_arms, p=self.P)  # 按概率选择动作
        self.Ps.append(self.P.copy())
        self.Qs.append(self.Q.copy())
        return action


if __name__ == "__main__":
    runs = 1
    steps = 200
    k_arms = 4
    np.random.seed(10)
    bandit = KAB_Softmax_test(k_arms, alpha=0.15)
    bandit.simulate(runs, steps)

    lines = ["-", "--", "-.", ":"]  # 线条风格
    grid = plt.GridSpec(nrows=1, ncols=2)
    plt.subplot(grid[0, 0])
    Q_array = np.array(bandit.Qs)
    for i in range(k_arms):
        plt.plot(Q_array[:,i], label=str(i+1), linestyle=lines[i])
    plt.title(u'备选动作的价值变化')
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'动作价值')
    plt.grid()
    plt.legend()
    print("最终的动作价值 =", np.round(Q_array[-1,:], 2))

    plt.subplot(grid[0, 1])
    P_array = np.array(bandit.Ps)
    for i in range(k_arms):
        plt.plot(P_array[:,i], label=str(i+1), linestyle=lines[i])
    plt.title(u'备选动作概率的变化')
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'被选概率')
    plt.grid()
    plt.legend()
    plt.show()
    
    print("最终的备选概率 =", P_array[-1,:])
    print("备选概率的和 =", np.sum(P_array[-1,:]))
