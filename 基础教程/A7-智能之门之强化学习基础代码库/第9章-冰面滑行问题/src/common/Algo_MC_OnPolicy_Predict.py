
from common.Algo_MC_OnPolicy_Base import MC_On_Policy_Base


# 首次访问法预测 V_pi
class MC_FirstVisit_Predict_V(MC_On_Policy_Base):
    def calculate(self):
        num_step = super().calculate()
        G = 0
        for t in range(num_step-1, -1, -1):
            s, _, G = super().calculate_G(G, t)
            if not (s in self.Trajectory_state[0:t]):  # 首次访问型
                self.CumV[s] += G     # 值累加
                self.CntV[s] += 1     # 数量加 1            
        
    def calculate_V(self):
        V, _ = super().calculate_V()  # 只需要 V，不需要 Q
        return V


# 首次访问法预测 V_pi,Q_pi
class MC_FirstVisit_Predict_VQ(MC_On_Policy_Base):
    def calculate(self):
        num_step = super().calculate()
        G = 0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)
            if not ((s,a) in self.Trajectory_sa[0:t]):  # 首次访问型
                super().increase_Cum_Count(s, a, G)     # 值累加，数量加 1


# 每次访问法预测 V_pi，Q_pi
class MC_EveryVisit_Predict_VQ(MC_On_Policy_Base):
    def calculate(self):
        num_step = super().calculate()
        G = 0
        for t in range(num_step-1, -1, -1):
            s, a, G = super().calculate_G(G, t)            
            super().increase_Cum_Count(s, a, G)     # 值累加，数量加 1
