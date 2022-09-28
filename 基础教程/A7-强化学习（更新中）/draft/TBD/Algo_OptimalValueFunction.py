import numpy as np
import copy

# 式 (8.4.2) 计算 q
def q_pi(p_s_r, gamma, V):
    q = 0
    # 遍历每个转移概率,以计算 q
    for p, s_next, r in p_s_r:
        # math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
        q += p * (r + gamma * V[s_next])
    return q

# 式 (8.4.5) 计算 v_pi
def v_pi(s, actions, gamma, V, Q):
    list_q = []
    for a, p_s_r in actions:        # 遍历每个动作以计算q值，进而计算v值
        q = q_pi(p_s_r, gamma, V)
        list_q.append(q)
        # 顺便记录下q(s,a)值,不需要再单独计算一次
        Q[s,a] = q
    return max(list_q)
           #V[s] = max(list_q) if len(list_q) > 0 else 0

def calculate_Vstar(env, gamma, max_iteration):
    V = np.zeros(env.nS)
    Q = np.zeros((env.nS, env.nA))
    count = 0
    # 迭代
    while (count < max_iteration):
        V_old = V.copy()
        # 遍历所有状态 s
        for s in range(env.nS):
            if env.is_end(s):   # 终止状态v=0
                continue            
            # 获得 状态->动作 策略概率
            actions = env.get_actions(s)
            V[s] = v_pi(s, actions, gamma, V, Q)
        # 检查收敛性
        if abs(V-V_old).max() < 1e-4:
            break
        count += 1
    # end while
    print("迭代次数 = ",count)
    return V, Q

def get_policy(env, V, gamma):
    policy = np.zeros((env.nS, env.nA))    
    for s in range(env.nS):
        actions = env.get_actions(s)
        list_q = []
        if actions is None:
            continue
        # 遍历每个策略概率
        for action, next_p_s_r in actions:
            q_star = 0
            for p, s_next, r in next_p_s_r:
                q_star += p * (r + gamma * V[s_next])
            list_q.append(q_star)
        policy[s, np.argmax(list_q)] = 1
    return policy

