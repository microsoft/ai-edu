
from common.Algo_TD_n_Base import TD_n_Base
import tqdm
import numpy as np

class TD_n(TD_n_Base):
    def run(self):
        for episode in range(self.episodes):
            done = False
            s, _ = self.env.reset()
            self.reward_Trajactor = [0]
            self.state_Trajectory = [s]
            T = float('inf')
            t = 0
            while True:
                if t < T:  # 采样到终止状态了，不继续采样，但是并不推出 while 循环
                    a = np.random.choice(self.env.action_space.n, p=self.behavior_policy[s])
                    s_next, reward, done, truncated, _ = self.env.step(a)
                    self.reward_Trajactor.append(reward)
                    self.state_Trajectory.append(s_next)
                    if done:  # 此时不退出 while 循环，而是仍要继续计算 n 次尾部数据
                        T = t + 1
                tau = t - self.n + 1
                if tau >= 0:  # 在前 n 次采样后，才开始更新 V，并且在 done=True 后，仍然计算 n 次，每次序列长度减一
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += pow(self.gamma, i - tau - 1) * self.reward_Trajactor[i]  # 如果初始化时没有给[0], 则此处应该是 i-1
                    
                    if tau + self.n < T:  # 最后一个状态V=0，不需要计算
                        G += pow(self.gamma, self.n) * self.V[self.state_Trajectory[tau + self.n]]
                    self.V[self.state_Trajectory[tau]] += self.alpha * (G - self.V[self.state_Trajectory[tau]])
                if tau == T - 1:  # 计算到只剩终止状态了，退出 while 循环
                    break            
                s = s_next
                t += 1
        return self.V
