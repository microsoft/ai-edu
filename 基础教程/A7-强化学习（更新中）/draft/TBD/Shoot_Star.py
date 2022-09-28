@@ -1,111 +0,0 @@
import numpy as np
from enum import Enum
import copy
import Shoot_2_DataModel as dataModel
import Algo_OptimalValueFunction as algo

'''
class Env(object):
    def __init__(self):
        self.nS = 31
        self.nA = 2
        self.P = P
        self.Policy = {0:0.4, 1:0.6}

    def reset(self, from_start = True):
        if (from_start):
            return self.States.Start.value
        else:
            idx = np.random.choice(self.state_space)
            return idx

    def get_actions(self, s):
        if (s < 6):
            actions = self.P[s]
            return actions.items()
        else:
            return None

    def get_states(self, a):
        for s, actions in self.P.items():
            if actions.__contains__(a):
                return actions[a]

    def step(self, curr_state, action):
        probs = self.P[curr_state][action]
        if (len(probs) == 1):
            return self.P[curr_state][action][0]
        else:
            idx = np.random.choice(3, p=self.transition)
            return self.P[curr_state][action][idx]



def V_star(env: Env, gamma):
    V_curr = [0.0] * env.nS
    V_next = [0.0] * env.nS
    Q_star = copy.deepcopy(env.P)    # 拷贝状态转移结构以存储Q(s,a)值
    count = 0
    # 迭代
    while (True):
        # 遍历所有状态 s
        for s in range(env.nS):
            # 获得 状态->动作 策略概率
            actions = env.get_actions(s)
            list_q = []
            if actions is not None:
                # 遍历每个策略概率
                for action, next_p_s_r in actions:
                    # 获得 动作->状态 转移概率
                    q_star = 0
                    # 遍历每个转移概率,以计算 q_pi
                    for p, s_next, r in next_p_s_r:
                        # 式2.1 math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
                        q_star += p * (r + gamma * V_curr[s_next])
                    #end for
                    # 式5 math: \sum_a \pi(a|s) q_\pi (s,a)
                    list_q.append(q_star)
                    Q_star[s][action] = q_star
                # end for
            V_next[s] = max(list_q) if len(list_q) > 0 else 0
        #endfor
        # 检查收敛性
        if np.allclose(V_next, V_curr):
            break
        # 把 V_curr 赋值给 V_next
        V_curr = V_next.copy()
        count += 1
    # end while
    print(count)
    print(Q_star)
    return V_next

def get_policy(env: Env, V, gamma):
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
'''

if __name__=="__main__":
    env = dataModel.Env(None)
    gamma = 1
    max_iteration = 100
    V_star, Q_star = algo.calculate_Vstar(env, gamma, max_iteration)
    print(np.round(V_star,4))
    policy = algo.get_policy(env, V_star, gamma)
    print(policy)
    print(Q_star)


