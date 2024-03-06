import numpy as np
import tqdm
import gymnasium as gym
from common.Algo_TD_Base import TD_Base

# Dyna-Q 算法
class DynaQ(TD_Base):
    def __init__(self, 
        env: gym.Env,           # 环境变量
        episodes: int,          # 采样幕数
        policy: np.ndarray,     # 输入策略        
        alpha: float = 0.1,     # 学习率
        gamma: float = 0.9,     # 折扣
        epsilon: float = 0.1,    # epsilon-greedy
        plan_step: int = 1,             # n-step TD
    ):
        super().__init__(env, episodes, policy, alpha=alpha, gamma=gamma, epsilon=epsilon)
        self.planning_steps = plan_step
        self.model = {}                         # 模型

    # def feed(self, state, action, reward, next_state):
    #     # if tuple(state) not in self.model.keys():
    #     if state not in self.model.keys():
    #         self.model[state] = dict()  # 创建一个子字典，以state为key
    #     self.model[state][action] = [next_state, reward]
    def feed(self, state, action, reward, next_state):        
        self.model[(state,action)] = (next_state, reward)

    # randomly sample from previous experience
    # def sample(self):
    #     state_index = np.random.choice(range(len(self.model.keys())))
    #     state = list(self.model)[state_index]
    #     action_index = np.random.choice(range(len(self.model[state].keys())))
    #     action = list(self.model[state])[action_index]
    #     next_state, reward = self.model[state][action]
    #     return state, action, next_state, reward

    def sample(self):
        state_action = list(self.model)[np.random.choice(range(len(self.model.keys())))]
        next_state, reward = self.model[state_action]
        return state_action[0], state_action[1], next_state, reward

    def run(self):
        for _ in range(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            done = False
            steps = 0
            while not done:  # 幕内采样
                steps += 1
                curr_action = self.choose_action(curr_state)    # 选择动作
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                # 式（10.4.1）q(s,a) <- q(s,a) + alpha * [r + gamma * Max[q(s')] - q(s,a)]
                self.Q[curr_state, curr_action] += \
                    self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[curr_state, curr_action])
                self.update_policy_average(curr_state)
                self.feed(curr_state, curr_action, reward, next_state)
                for _ in range(self.planning_steps):
                    state_, action_, next_state_, reward_ = self.sample()
                    self.Q[state_, action_] += \
                        self.alpha * (reward_ + self.gamma * np.max(self.Q[next_state_, :]) - self.Q[state_, action_])
                    self.update_policy_average(state_)
                curr_state = next_state
            # end while
        #end for
            print("steps: %d" % steps)
            yield self.Q
        # return self.Q
