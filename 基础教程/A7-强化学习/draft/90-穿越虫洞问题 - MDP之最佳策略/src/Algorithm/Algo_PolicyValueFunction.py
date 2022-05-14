import numpy as np

# 式 (8.4.2) 计算 q_pi
def q_pi(p_s_r, gamma, V):
    q = 0
    # 遍历每个转移概率,以计算 q_pi
    for p, s_next, r in p_s_r:
        # math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
        q += p * (r + gamma * V[s_next])
    return q

# 式 (8.4.5) 计算 v_pi
def v_pi(policy, s, actions, gamma, V, Q):
    v = 0
    for a, p_s_r in actions:        # 遍历每个动作以计算q值，进而计算v值
        q = q_pi(p_s_r, gamma, V)
        # math: \sum_a \pi(a|s) q_\pi (s,a)
        v += policy[s][a] * q
        # 顺便记录下q(s,a)值,不需要再单独计算一次
        Q[s,a] = q
    return v

# 迭代法计算 v_pi
def calculate_VQ_pi(env, gamma, iteration):
    V = np.zeros(env.nS)            # 初始化 V(s)
    Q = np.zeros((env.nS, env.nA))  # 初始化 Q(s,a)
    count = 0   # 计数器，用于衡量性能和避免无限循环
    # 迭代
    while (count < iteration):
        V_old = V.copy()    # 保存上一次的值以便检查收敛性
        # 遍历所有状态 s
        for s in range(env.nS):
            if env.is_end(s):   # 终止状态v=0
                continue
            actions = env.get_actions(s)    # 获得当前状态s下的所有可选动作
            V[s] = v_pi(env.Policy, s, actions, gamma, V, Q)
        # 检查收敛性
        if abs(V-V_old).max() < 1e-4:
            break
        count += 1
    # end while
    print("迭代次数 = ",count)
    return V, Q
