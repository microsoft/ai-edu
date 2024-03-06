
import numpy as np
import gymnasium as gym
import common.Algo_MC_OffPolicy_Base as OffPolicyBase


# MC 离轨控制 重要性采样 + 每次访问法 估计 PI*
# WIS = Weighted Importance Sampling 加权重要性采样
class MC_OffPolicy_Control_Policy(OffPolicyBase.MC_OffPolicy_Base):
    def calculate(self):        
        num_step = super().calculate()
        G = 0.0
        W = 1.0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)
            if not ((s,a) in self.Trajectory_sa[0:t]):  # 首次访问型
                self.Qcum[s,a] += W * G # 值累加
                self.Qcount[s,a] += W     # 量累加
                self.Q[s,a] = self.Qcum[s,a] / self.Qcount[s,a]
                # 更新目标策略
                best_a = super().update_target_policy(s)
                if best_a != a: # 如果不是最优动作，就跳出循环
                    break
                # best_actions = super().update_target_policy_average(s, round=1)
                # if a not in best_actions:
                #     break
                # 更新权重
                W = W  / self.behavior_policy[s, a]

    def calculate_V(self):
        self.Qcount[self.Qcount==0] = 1 # 把分母为0的填成 1，主要是终止状态
        self.Q = self.Qcum / self.Qcount   # 求均值得到 Q
        return self.Q, self.target_policy


# MC 离轨控制 重要性采样 + 每次访问法 估计 Q* 增量版
class MC_OffPolicy_Control_Policy_Increamental(OffPolicyBase.MC_OffPolicy_Base):
    def calculate(self):        
        num_step = super().calculate()
        G = 0.0
        W = 1.0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)
            self.Qcount[s,a] += W     # 量累加
            self.Q[s,a] = self.Q[s,a] + W / self.Qcount[s,a] * (G - self.Q[s,a])
            # 更新目标策略
            # best_actions = np.argwhere(self.Q[s] == np.max(self.Q[s]))
            # best_actions_count = len(best_actions)
            # self.target_policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(self.nA)]
            best_a = np.argmax(self.Q[s])
            self.target_policy[s] = 0.
            self.target_policy[s, best_a] = 1.
            if best_a != a: # 如果不是最优动作，就跳出循环
                break
            # 更新权重
            W = W  / self.behavior_policy[s, a]

    def calculate_V(self):
        return self.Q, self.target_policy


# MC 离轨控制 重要性采样 + 每次访问法 估计 PI*
# NIS = Normal Importance Sampling 普通重要性采样
class MC_OffPolicy_NIS_Control_Policy(OffPolicyBase.MC_OffPolicy_Base):
    def calculate(self):        
        num_step = super().calculate()
        G = 0.0
        W = 1.0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)
            self.Qcum[s,a] += W * G # 值累加
            self.Qcount[s,a] += 1     # 量累加
            # self.Q[s,a] = self.CumQ[s,a] / self.CntQ[s,a]
            # 更新目标策略
            # best_actions = np.argwhere(self.Q[s] == np.max(self.Q[s]))
            # best_actions_count = len(best_actions)
            # self.target_policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(self.nA)]
            #best_a = np.argmax(self.Q[s])
            #self.target_policy[s] = 0.
            #self.target_policy[s, best_a] = 1.
            #if best_a != a: # 如果不是最优动作，就跳出循环
            #    break
            # 更新权重
            W = W  / self.behavior_policy[s, a]

    def calculate_V(self):
        self.Qcount[self.Qcount==0] = 1 # 把分母为0的填成 1，主要是终止状态
        self.Q = self.Qcum / self.Qcount   # 求均值得到 Q
        for s in range(self.nS):
            best_actions = np.argwhere(self.Q[s] == np.max(self.Q[s]))
            best_actions_count = len(best_actions)
            self.target_policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(self.nA)]
        return self.Q, self.target_policy
