# 策略迭代算法
import copy
import numpy as np
import common.Algo_PolicyEvaluation as PolicyEvaluation

# 根据上一次的策略计算出来的 V 值，计算新的策略
def policy_improvement_V(env, V, gamma):
    policy = copy.deepcopy(env.Policy)  # dict {0:[...], 1:[...], ...}
    for s in range(env.nS):  # 遍历状态
        actions = env.get_actions(s)  # 获得当前状态s下的所有可选动作的(概率、下一状态、奖励)=(p, s', r)
        # 需要先根据 V 计算 Q
        Q_s = [0] * env.nA
        if actions is None:
            continue
        # 遍历每个策略概率计算 Q 值
        for action, next_p_s_r in actions:  # dict {0:[(0.8, 0, -1), (0.1, 1, -1), (0.1, 2, -1)], ...}
            q = 0
            for p, s_next, r in next_p_s_r:  # (0.8, 0, -1)
                q += p * (r + gamma * V[s_next])
            Q_s[action] = q
        # best_action = np.argmax(Q)
        best_actions = np.argwhere(Q_s == np.max(Q))        
        best_actions_count = len(best_actions)
        policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(env.nA)]
    return policy    


# 根据上一次的策略计算出来的 Q 值，计算新的策略
def policy_improvement_Q(env, Q):
    policy = copy.deepcopy(env.Policy)  # 要求 env.Policy 是一个dict {s0:[a0,a1,...], s1:[...], ...}
    for s in range(env.nS):  # 遍历状态
        # 有多个相同值则均分，如 [0.5,0.5,0,0] or [0.25,0.25,0.25,0.25]
        best_actions = np.argwhere(Q[s] == np.max(Q[s]))
        best_actions_count = len(best_actions)
        policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(env.nA)]
    return policy    


# 策略迭代
def policy_iteration(env, gamma: float, max_iteration: int = 1000, verbose: bool = False):
    count = 0
    while True:
        print("策略评估")
        V, Q = PolicyEvaluation.calculate_VQ_pi(env, gamma, max_iteration)
        print("策略改进")
        # new_policy = policy_improvement_V(env, V, gamma)
        new_policy = policy_improvement_Q(env, Q)
        if new_policy == env.Policy:
            break
        else:
            if verbose:
                print(new_policy)
            env.Policy = new_policy.copy()
        count += 1
    print("策略迭代完成，共迭代 {} 次".format(count))  
    return env.Policy, V, Q
