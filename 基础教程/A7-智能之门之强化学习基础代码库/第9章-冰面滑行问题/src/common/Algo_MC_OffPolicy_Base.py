
import numpy as np
import tqdm
import gymnasium as gym

# 基类
class MC_OffPolicy_Base(object):
    # 初始化，可重载
    def __init__(
        self, 
        env: gym.Env, 
        episodes: int, 
        gamma: float, 
        behavior_policy: np.ndarray,
        target_policy: np.ndarray = None,
    ):
        self.nS = env.observation_space.n       # 状态空间的大小
        self.nA = env.action_space.n            # 动作空间的大小
        self.Vcum = np.zeros(self.nS, dtype=np.float32)           # 累加 G 的总和计算 V
        self.Vcount = np.zeros(self.nS, dtype=np.float32)           # 记录 G 值的次数计算 V
        self.Qcum = np.zeros((self.nS, self.nA), dtype=np.float32)# 累加 G 的总和计算 Q
        self.Qcount = np.zeros((self.nS, self.nA), dtype=np.float32)# 记录 G 值的次数计算 Q
        self.V = np.zeros(self.nS, dtype=np.float32)   # V 值 
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float32)   # Q 值 
        self.behavior_policy = behavior_policy  # 行为策略
        if target_policy is None:
            self.target_policy = np.zeros_like(behavior_policy)      # 目标策略
        else:
            self.target_policy = target_policy
        self.gamma = gamma                      # 折扣
        self.episodes = episodes                # 采样幕数
        self.env = env                          # 环境变量
        self.Trajectory_reward = []             # 记录时刻 t 的 reward
        self.Trajectory_sa = []                 # 记录时刻 t 的 (s,a) 对

    # 幕内采样，除非有特殊需要，否则不建议重载
    def sampling(self):
        state, _ = self.env.reset()         # 重置环境，开始新的一幕采样
        done = False                        # 一幕结束标志
        self.Trajectory_reward = []         # 清空 r 轨迹信息
        self.Trajectory_sa = []             # 清空 s,a 轨迹信息
        while (done is False):              # 幕内循环
            # 在当前状态 s 下根据策略 policy 选择一个动作
            action = np.random.choice(self.nA, p=self.behavior_policy[state])
            # 得到下一步的状态、奖励、结束标志、是否超限终止等
            next_state, reward, done, truncated, _ = self.env.step(action)
            if truncated:
                return False
            # 记录轨迹信息
            self.Trajectory_reward.append(reward)
            self.Trajectory_sa.append((state, action))
            state = next_state              # goto next state
        # if self.Trajectory_reward[-1] == 0:
        #     return False
        return True

    # 每幕结束后计算相关值
    # 在子类中重载并调用 num_step = super().calculate() 实现具体算法
    # 比如首次访问法和每次访问法，以及其它特殊的算法
    def calculate(self):
        assert(len(self.Trajectory_reward) == len(self.Trajectory_sa))
        return len(self.Trajectory_reward)

    # 根据上一次的 G 和本次的 r 计算新 G 值，需要在子类中调用，一般不需要重载
    def calculate_G(self, G, t):
        s, a = self.Trajectory_sa[t]
        r = self.Trajectory_reward[t]
        G = self.gamma * G + r
        return s, a, G

    # 所有幕结束后计算平均值
    def calculate_V(self):
        self.Vcount[self.Vcount==0] = 1     # 把分母为0的填成 1，主要是终止状态
        self.V = self.Vcum / self.Vcount       # 求均值得到 Q
        self.Qcount[self.Qcount==0] = 1     # 把分母为0的填成 1，主要是终止状态
        self.Q = self.Qcum / self.Qcount       # 求均值得到 Q
        return self.V, self.Q

    # 多幕循环与流程控制，不需要重载
    def run(self):
        for _ in tqdm.trange(self.episodes): # 多幕循环
            if not self.sampling():     # 一幕采样
                continue
            self.calculate()    # 一幕结束，进行必要的计算
        return self.calculate_V()

    # 更新目标策略（如有相等值取第一个），需要在子类中调用，一般不需要重载
    def update_target_policy(self, s):
        best_a = np.argmax(self.Q[s])
        self.target_policy[s] = 0.          # 先清空所有
        self.target_policy[s, best_a] = 1.  # 再设置最优动作
        return best_a
    
    # 更新目标策略（如有相等则平分）
    def update_target_policy_average(self, s, round=None):
        if round is not None:
            Qs = np.round(self.Q[s], round)
        else:
            Qs = self.Q[s]
        best_actions = np.argwhere(Qs == np.max(Qs))
        best_actions_count = len(best_actions)
        self.target_policy[s] = [1/best_actions_count if a in best_actions else 0 for a in range(self.nA)]
        return best_actions
