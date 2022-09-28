import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False


def draw_one_arm(reward_dist):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes[0]
    ax.grid()
    ax.hist(reward_dist, bins=21)

    ax = axes[1]
    ax.grid()
    ax.violinplot(reward_dist, showmeans=True, quantiles=[0,0.025,0.25,0.75,0.925])

    plt.show()

def draw_mu(reward_mu):
    plt.plot(reward_mu, 'ro--')
    plt.xlabel(u"10臂")
    plt.ylabel(u"期望均值")
    plt.show()

def draw_k_arm(k_reward_dist_mu, k_reward_dist_mu_sort):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = axes[0]
    ax.grid()
    ax.violinplot(k_reward_dist_mu, showmeans=True)
    mean = np.round(np.mean(k_reward_dist_mu, axis=0), 3)
    for i in range(10):
        ax.text(i+1+0.2,mean[i]-0.1,str(mean[i]))

    ax = axes[1]
    ax.grid()
    ax.violinplot(k_reward_dist_mu_sort, showmeans=True)
    mean = np.round(np.mean(k_reward_dist_mu_sort, axis=0), 3)
    for i in range(10):
        ax.text(i+1+0.2,mean[i]-0.1,str(mean[i]))

    plt.show()

if __name__=="__main__":
    # 生成原始数据
    num_arm = 10
    num_data = 2000
    np.random.seed(5)
    k_reward_dist = np.random.randn(num_data, num_arm)
    print("原始均值=", np.round(np.mean(k_reward_dist, axis=0),3))
    draw_one_arm(k_reward_dist[:,0])
    # 生成期望均值
    reward_mu = np.random.randn(num_arm)
    print("期望平均回报=", np.round(reward_mu,3))
    draw_mu(reward_mu)
    # 生成期望数据（=原始数据+期望均值）
    k_reward_dist_mu = reward_mu + k_reward_dist
    print("实际均值=", np.round(np.mean(k_reward_dist_mu, axis=0),3))
    # 按均值排序
    reward_mu_sort_arg = np.argsort(reward_mu)  # 对期望均值排序（并不实际排序，而是返回序号）
    k_reward_dist_mu_sort = np.zeros_like(k_reward_dist_mu)
    for i in range(10):
        idx = reward_mu_sort_arg[i] # 第i个臂对应的新序号是idx
        k_reward_dist_mu_sort[:,i] = k_reward_dist_mu[:,idx] # 重新排序
    draw_k_arm(k_reward_dist_mu, k_reward_dist_mu_sort)
