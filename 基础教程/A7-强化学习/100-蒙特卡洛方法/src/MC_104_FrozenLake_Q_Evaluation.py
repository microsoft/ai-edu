from cProfile import label
from email import policy
import numpy as np
import gym
import Algorithm.Algo_MonteCarlo_MDP as algoMC
import Algorithm.Algo_PolicyValueFunction as algoDP
import common.DrawQpi as drawQ
import common.CommonHelper as helper
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False

# MC 策略评估（预测）：每次访问法估算 Q_pi
def MC_EveryVisit_Q_Policy_test(env, episodes, gamma, policy, checkpoint=1000):
    nA = env.action_space.n
    nS = env.observation_space.n
    Value = np.zeros((nS, nA))  # G 的总和
    Count = np.zeros((nS, nA))  # G 的数量
    Q_history = []
    for episode in tqdm.trange(episodes):   # 多幕循环
        # 重置环境，开始新的一幕采样
        s, _ = env.reset(return_info=True)
        Episode = []     # 一幕内的(状态,奖励)序列
        done = False
        while (done is False):            # 幕内循环
            action = np.random.choice(nA, p=policy[s])
            next_s, reward, done, info = env.step(action)
            Episode.append((s, action, reward))
            s = next_s
        
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(len(Episode)-1, -1, -1):
            s, a, r = Episode[t]
            G = gamma * G + r
            Value[s,a] += G     # 值累加
            Count[s,a] += 1     # 数量加 1

        # 检查是否收敛
        if (episode + 1)%checkpoint == 0: 
            Count[Count==0] = 1 # 把分母为0的填成1，主要是对终止状态
            Q = Value / Count
            Q_history.append(Q)
    return Q_history

def get_groud_truth(env, policy, gamma):
    iteration = 100
    _, Q = algoDP.calculate_VQ_pi(env, policy, gamma, iteration)
    return Q

def create_policy(env, args):
    left = args[0]
    down = args[1]
    right = args[2]
    up = args[3]
    assert(left+down+right+up==1)
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    policy[:, 0] = left
    policy[:, 1] = down
    policy[:, 2] = right
    policy[:, 3] = up
    return policy

if __name__=="__main__":
    gamma = 1
    episodes = 50000
    policy_names = ["正确方向", "随机方向","错误方向"]
    policies = [
        # left, down, right, up
        (0.2,  0.3,  0.3,  0.2),
        (0.25, 0.25, 0.25, 0.25), 
        (0.3,  0.2,  0.2,  0.3)
    ]
    np.set_printoptions(suppress=True)
    for i, policy_data in enumerate(policies):
        env = gym.make("FrozenLake-v1", desc=None, map_name = "4x4", is_slippery=False)
        policy = create_policy(env, policy_data)
        print(policy)
        Q_real = get_groud_truth(env, policy, gamma)
        nA = env.action_space.n
        nS = env.observation_space.n
        start_state, info = env.reset(seed=5, return_info=True)
        #Q_history = algoMC.MC_EveryVisit_Q_Policy(env, start_state, episodes, gamma, policy)
        Q_history = MC_EveryVisit_Q_Policy_test(env, episodes, gamma, policy)
        Errors = []
        for Q in Q_history:
            error = helper.RMSE(Q, Q_real)
            Errors.append(error)

        print("------ 状态价值函数 -----")
        print(np.round(Q,3))
        print("误差 =", error)
        plt.plot(Errors, label=policy_names[i])

        #Q4 = np.round(Q,4)
        #print(Q)
        drawQ.draw(Q,(4,4))
        env.close()
        print(helper.RMSE(Q, Q_real))

    plt.title(u'策略评估 $V_\pi$ 的误差与循环次数的关系')
    plt.xlabel(u'循环次数(x1000)')
    plt.ylabel(u'误差 RMSE')
    plt.legend()
    plt.grid()
    plt.show()
