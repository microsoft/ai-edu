import tqdm
import gymnasium as gym
import numpy as np
from common.Algo_TD_Base import TD_Base
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as ss


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False

# Sarsa 算法
class TD_SARSA(TD_Base):
    def choose_action(self, state): # 选择动作，从行为策略中按概率选择
        action = np.random.choice(self.nA, p=self.behavior_policy[state])
        return action

    def update_policy_max(self, state): # 按e-greedy策略更新行为策略
        best_action = np.argmax(self.Q[state])
        self.behavior_policy[state] = self.epsilon/(self.nA-1)
        self.behavior_policy[state, best_action] = 1 - self.epsilon

    def run(self):
        R = np.zeros(self.episodes)
        for episode in range(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            curr_action = self.choose_action(curr_state)        # 选择动作
            done = False
            r = 0
            while not done:  # 幕内采样
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                R[episode] += reward
                next_action = self.choose_action(next_state)    # 选择动作
                # 式（10.3.4）：q(s,a) <- q(s,a) + alpha * [r + gamma * q(s',a') - q(s,a)]
                self.Q[curr_state, curr_action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[curr_state, curr_action])
                self.update_policy_max(curr_state)                  # 更新行为策略
                curr_state = next_state
                curr_action = next_action
            #endwhile
        #endfor          
        return R


# 期望 Sarsa
class TD_E_SARSA(TD_SARSA):
    def expected_Q(self, state): # 计算期望 Q
        v = np.dot(self.Q[state], self.behavior_policy[state])
        return v

    # E_SARSA 算法
    def run(self):
        R = np.zeros(self.episodes)
        for episode in range(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            curr_action = self.choose_action(curr_state)        # 选择动作
            done = False
            step = 0
            while not done:  # 幕内采样
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                R[episode] += reward
                next_action = self.choose_action(next_state)    # 选择动作
                # 式（10.4.1）q(s,a) <- q(s,a) + alpha * [r + gamma * q(s',a') - q(s,a)]
                expected = self.expected_Q(next_state)
                self.Q[curr_state, curr_action] += self.alpha * (reward + self.gamma * expected - self.Q[curr_state, curr_action])
                self.update_policy_max(curr_state)                  # 更新行为策略
                curr_state = next_state
                curr_action = next_action
            #endwhile
        #endfor
        return R


# Q-Learning 算法
class TD_QLearning(TD_SARSA):
    def run(self):
        R = np.zeros(self.episodes)
        for episode in range(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            done = False
            while not done:  # 幕内采样
                curr_action = self.choose_action(curr_state)    # 选择动作
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                R[episode] += reward
                # 式（10.4.1）q(s,a) <- q(s,a) + alpha * [r + gamma * Max[q(s')] - q(s,a)]
                self.Q[curr_state, curr_action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[curr_state, curr_action])
                self.update_policy_max(curr_state)                  # 更新行为策略
                curr_state = next_state
            #endwhile
        #endfor      
        return R


# Double Q-Learning 算法
class TD_DQLearning(TD_SARSA):
    def __init__(self, env, episodes, policy, alpha, gamma, epsilon):
        super().__init__(env, episodes, policy, alpha, gamma, epsilon)
        self.Q1 = self.Q                        # 初始化第一个Q表
        self.Q2 = np.zeros((self.nS, self.nA))  # 初始化第二个Q表

    # 两个相同价值的动作只选择第一个
    def update_policy_max(self, state): # 按e-greedy策略更新行为策略
        best_action = np.argmax(self.Q1[state] + self.Q2[state])
        self.behavior_policy[state] = self.epsilon/(self.nA-1)
        self.behavior_policy[state, best_action] = 1 - self.epsilon

    def run(self):
        R = np.zeros(self.episodes)
        id = 1  # 第一次用Q1
        for episode in range(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            done = False
            while not done:  # 幕内采样
                curr_action = self.choose_action(curr_state)    # 选择动作
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                R[episode] += reward
                # 式（10.5.1）q1(s,a) <- q1(s,a) + alpha * [r + gamma * q2(s',argmax(q1(s',:))) - q1(s,a)]
                if id == 1:
                    id = 2  # 下一次用Q2
                    best_action = np.argmax(self.Q1[next_state])
                    self.Q1[curr_state, curr_action] += self.alpha * (reward + self.gamma * self.Q2[next_state, best_action] - self.Q1[curr_state, curr_action])
                else:  # assert id == 2
                    id = 1  # 下一次用Q1
                    best_action = np.argmax(self.Q2[next_state])
                    self.Q2[curr_state, curr_action] += self.alpha * (reward + self.gamma * self.Q1[next_state, best_action] - self.Q2[curr_state, curr_action])
                self.update_policy_max(curr_state)                  # 更新行为策略
                curr_state = next_state
        return R


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    env.reset(seed=5)
    Episodes = 500 # 500
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    R = np.zeros((4, Episodes))
    runs = 50 # 50
    for run in tqdm.trange(runs):
        ctrl = TD_SARSA(env, Episodes, behavior_policy, alpha, gamma, epsilon)
        R[0] += ctrl.run()
        ctrl = TD_E_SARSA(env, Episodes, behavior_policy, alpha, gamma, epsilon)
        R[1] += ctrl.run()
        ctrl = TD_QLearning(env, Episodes, behavior_policy, alpha, gamma, epsilon)
        R[2] += ctrl.run()
        ctrl = TD_DQLearning(env, Episodes, behavior_policy, alpha, gamma, epsilon)
        R[3] += ctrl.run()

    start_episode = 25
    labels = ["Sarsa", "E-Sarsa", "Q-Learning", "Double Q"]
    lines = ["-", "--", ":", "-."]
    for i in range(4):
        tmp = ss.savgol_filter(R[i,start_episode:]/runs, 40, 2)
        plt.plot(tmp, label=labels[i], linestyle=lines[i])
    plt.legend()
    plt.grid()
    plt.xlabel("幕")
    plt.ylabel("每幕的回报")
    plt.show()
