
import numpy as np
import tqdm
import gymnasium as gym

# 基类
class MC_On_Policy_Base(object):
    # 初始化，可重载
    def __init__(self, 
        env: gym.Env, 
        episodes: int,
        gamma: float, 
        policy: np.ndarray
    ):
        self.nS = env.observation_space.n       # 状态空间的大小
        self.nA = env.action_space.n            # 动作空间的大小
        self.CumV = np.zeros(self.nS, dtype=np.float32)           # 累加 G 的总和计算 V
        self.CntV = np.zeros(self.nS, dtype=np.float32)           # 记录 G 值的次数计算 V
        self.CumQ = np.zeros((self.nS, self.nA), dtype=np.float32)# 累加 G 的总和计算 Q
        self.CntQ = np.zeros((self.nS, self.nA), dtype=np.float32)# 记录 G 值的次数计算 Q
        self.policy = policy                    # 输入策略
        self.gamma = gamma                      # 折扣
        self.episodes = episodes                # 采样幕数
        self.env = env                          # 环境变量
        self.Trajectory_reward = []             # 记录时刻 t 的 reward
        self.Trajectory_sa = []                 # 记录时刻 t 的 (s,a) 对
        self.Trajectory_state =  []             # 记录时刻 t 的 (s)

    # 幕内采样，除非有特殊需要，否则不建议重载
    def sampling(self):
        state, _ = self.env.reset()         # 重置环境，开始新的一幕采样
        done = False                        # 一幕结束标志
        self.Trajectory_reward = []         # 清空 r 轨迹信息
        self.Trajectory_sa = []             # 清空 s,a 轨迹信息
        self.Trajectory_state = []          # 清空 s 轨迹信息
        while (done is False):              # 幕内循环
            # 在当前状态 s 下根据策略 policy 选择一个动作
            action = np.random.choice(self.nA, p=self.policy[state])
            # 得到下一步的状态、奖励、结束标志、是否超限终止等
            next_state, reward, done, truncated, _ = self.env.step(action)
            # 记录轨迹信息
            self.Trajectory_reward.append(reward)
            self.Trajectory_sa.append((state, action))
            self.Trajectory_state.append(state)
            state = next_state              # goto next state

    # 每幕结束后计算相关值
    # 在子类中重载并调用 num_step = super().calculate() 实现具体算法
    # 比如首次访问法和每次访问法，以及其它特殊的算法
    def calculate(self):
        assert(len(self.Trajectory_reward) == len(self.Trajectory_sa) == len(self.Trajectory_state))
        return len(self.Trajectory_reward)

    # 根据上一次的 G 和本次的 r 计算新 G 值，需要在子类中调用，一般不需要重载
    def calculate_G(self, G, t):
        s, a = self.Trajectory_sa[t]
        r = self.Trajectory_reward[t]
        G = self.gamma * G + r
        return s, a, G

    # 增加累加值和计数值的帮助函数，需要在子类中调用
    def increase_Cum_Count(self, s, a, G):
        self.CumV[s] += G     # 值累加
        self.CntV[s] += 1     # 数量加 1            
        self.CumQ[s,a] += G     # 值累加
        self.CntQ[s,a] += 1     # 数量加 1            
        
    # 所有幕结束后计算平均值
    def calculate_V(self):
        self.CntV[self.CntV==0] = 1     # 把分母为0的填成 1，主要是终止状态
        V = self.CumV / self.CntV       # 求均值得到 V
        self.CntQ[self.CntQ==0] = 1     # 把分母为0的填成 1，主要是终止状态
        Q = self.CumQ / self.CntQ       # 求均值得到 Q
        return V, Q

    # 多幕循环与流程控制，不需要重载
    def run(self):
        for _ in tqdm.trange(self.episodes): # 多幕循环
            self.sampling()     # 一幕采样
            self.calculate()    # 一幕结束，进行必要的计算
        return self.calculate_V()
