import numpy as np

# 式 (8.4.2) 计算 q_pi
def q_pi(p_s_r_d, gamma, V):
    q = 0
    # 遍历每个转移概率,以计算 q_pi
    for p, s_next, reward, done in p_s_r_d:
        # math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
        q += p * (reward + gamma * V[s_next])
    return q

# 式 (8.4.5) 计算 v_pi
def v_pi(policy, s, actions, gamma, V, Q):
    v = 0
    for action in list(actions.keys()):  # actions 是一个字典数据，key 是动作
        q = q_pi(actions[action], gamma, V)
        # math: \sum_a \pi(a|s) q_\pi (s,a)
        v += policy[s][action] * q
        # 顺便记录下q(s,a)值,不需要再单独计算一次
        Q[s,action] = q
    return v

# 迭代法计算 v_pi
def calculate_VQ_pi(env, policy, gamma, iteration, delta=1e-4):
    V = np.zeros(env.observation_space.n)            # 初始化 V(s)
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # 初始化 Q(s,a)
    count = 0   # 计数器，用于衡量性能和避免无限循环
    # 迭代
    while (count < iteration):
        V_old = V.copy()    # 保存上一次的值以便检查收敛性
        # 遍历所有状态 s
        for s in range(env.observation_space.n):
            actions = env.P[s]
            V[s] = v_pi(policy, s, actions, gamma, V, Q)
        # 检查收敛性
        if abs(V-V_old).max() < delta:
            break
        count += 1
    # end while
    print("迭代次数 = ",count)
    return V, Q
