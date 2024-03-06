
import numpy as np
import gymnasium as gym
import common.Algo_MC_OffPolicy_Base as OffPolicyBase

# MC 离轨预测 重要性采样 + 每次访问法 估计 V_pi, Q_pi
class MC_OffPolicy_Predict_VQ(OffPolicyBase.MC_OffPolicy_Base):
    def calculate(self):        
        num_step = super().calculate()
        G = 0.0
        W = 1.0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)
            if not ((s,a) in self.Trajectory_sa[0:t]):  # 首次访问型
                self.Qcount[s,a] += W    
                self.Qcum[s,a] += W * G
                W = W * self.target_policy[s][a] / self.behavior_policy[s][a]
                self.Vcount[s] += W
                self.Vcum[s] += W * G
                if W == 0:
                    break


# 预测 Q* 增量版
class MC_OffPolicy_Predict_Q_Increamental(OffPolicyBase.MC_OffPolicy_Base):
    def calculate(self):        
        num_step = super().calculate()
        G = 0.0
        W = 1.0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)
            self.Qcount[s,a] += W    
            self.Q[s,a] = self.Q[s,a] + W  / self.Qcount[s,a] * (G - self.Q[s,a])
            W = W * self.target_policy[s, a] / self.behavior_policy[s, a]
            if W == 0:
                break

    def calculate_V(self):
        return self.Q
