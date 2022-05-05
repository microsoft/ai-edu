import numpy as np

# 式 (2.1) 计算 q_pi
def q_pi(p_s_r, gamma, V):
    q = 0
    # 遍历每个转移概率,以计算 q_pi
    for p, s_next, r in p_s_r:
        # math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
        q += p * (r + gamma * V[s_next])
    return q

# 式 (5) 计算 v_pi
def v_pi(env, s, gamma, V, Q):
    if env.is_end(s):
        return 0
    actions = env.get_actions(s)    # 获得当前状态s下的所有可选动作
    v = 0
    for a, p_s_r in actions:        # 遍历每个动作以计算q值，进而计算v值
        q = q_pi(p_s_r, gamma, V)
        # math: \sum_a \pi(a|s) q_\pi (s,a)
        v += env.Policy[s][a] * q
        # 顺便记录下q(s,a)值,不需要再单独计算一次
        Q[s,a] = q
    return v

# 迭代法计算 v_pi
def V_in_place_update(env, gamma, iteration):
    V = np.zeros(env.nS)            # 初始化 V(s)
    Q = np.zeros((env.nS, env.nA))  # 初始化 Q(s,a)
    count = 0   # 计数器，用于衡量性能和避免无限循环
    # 迭代
    while (count < iteration):
        V_old = V.copy()    # 保存上一次的值以便检查收敛性
        # 遍历所有状态 s
        for s in range(env.nS):
            V[s] = v_pi(env, s, gamma, V, Q)
        # 检查收敛性
        if abs(V-V_old).max() < 1e-4:
            break
        count += 1
    # end while
    print("迭代次数 = ",count)
    return V, Q

# 双数组迭代
def V_pi_2array(env, gamma, iteration):
    V = np.zeros(env.nS)
    Q = np.zeros((env.nS, env.nA))
    count = 0
    # 迭代
    while (count < iteration):
        V_old = V.copy()
        # 遍历所有状态 s
        for s in range(env.nS):
            v_pi = 0
            # 获得 状态->动作 策略概率
            actions = env.get_actions(s)
            if actions is not None:
                # 遍历每个策略概率

                for action, next_p_s_r in actions:
                    # 获得 动作->状态 转移概率
                    q_pi = 0
                    # 遍历每个转移概率,以计算 q_pi
                    for p, s_next, r in next_p_s_r:
                        # 式2.1 math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
                        q_pi += p * (r + gamma * V_old[s_next])
                    #end for
                    # 式5 math: \sum_a \pi(a|s) q_\pi (s,a)
                    Q[s][action] = q_pi
                    v_pi += 0.25 * q_pi
                # end for
            V[s] = v_pi
        #endfor
        # 检查收敛性
        if abs(V-V_old).max() < 1e-4:
            break
        # 把 V_curr 赋值给 V_next
        #print(np.reshape(np.round(V,2), (4,4)))
        count += 1
    # end while
    print(count)
    #print(Q)
    return V, Q
