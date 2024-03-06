
import common.CommonHelper as helper
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from common.Algo_TD_SARSA import TD_SARSA
import numpy as np
import gymnasium as gym
import tqdm
from inspect import isfunction
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False


# 状态空间
class States(Enum):
     S = 0  # 开始
     M = 1  # 中间
     T = 2  # 终止

def reward_func():
    return np.random.normal(-0.1, 1)

P = {
    # state: {action: [(p, s', r),...]}
    States.S.value:{                                
        0:[(1.0, States.T.value, 0, True),],       # left
        1:[(1.0, States.M.value, 0, False),]       # right       
    },
    States.M.value:{ 
        0:[(1.0, States.T.value, reward_func, True),],
        1:[(1.0, States.T.value, reward_func, True),],
        2:[(1.0, States.T.value, reward_func, True),],
        3:[(1.0, States.T.value, reward_func, True),],
        4:[(1.0, States.T.value, reward_func, True),],
        5:[(1.0, States.T.value, reward_func, True),],
        6:[(1.0, States.T.value, reward_func, True),],
        7:[(1.0, States.T.value, reward_func, True),],
        8:[(1.0, States.T.value, reward_func, True),],
        9:[(1.0, States.T.value, reward_func, True),],
    },
    States.T.value:{     
        0:[(1.0, States.T.value, 0, True),],
    }
}

class Env(object):
    def __init__(self):
        self.observation_space = gym.spaces.Discrete(len(States))
        self.action_space = gym.spaces.Discrete(10)
        self.nA = [2,10]
        self.S = States
        self.P = P
        self.end_states = [States.T.value]

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.state = States.S.value
        return States.S.value, 0

    def step(self, a):
        list_PSRE = self.P[self.state][a]
        list_P = [item[0] for item in list_PSRE]
        index = np.random.choice(len(list_PSRE), p=list_P)
        p, s_next, r, is_end = list_PSRE[index]
        self.state = s_next
        if isfunction(r):
            r = r()
        return s_next, r, is_end, None, None


class TD_QLearning(TD_SARSA):
    def __init__(self, env, episodes, policy, alpha, gamma, epsilon):
        super().__init__(env, episodes, policy, alpha, gamma, epsilon)
        self.nA = [2,10]
        self.Q = {
            0: np.zeros(2),
            1: np.zeros(10),
            2: 0
        }

    def run(self):
        wrong_count = np.zeros(self.episodes)
        for episode in range(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            done = False
            while not done:  # 幕内采样
                curr_action = self.choose_action(curr_state)    # 选择动作
                if curr_state == 0 and curr_action == 1: # right to S_M
                    wrong_count[episode] = 1
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                # 式（10.4.1）q(s,a) <- q(s,a) + alpha * [r + gamma * Max[q(s')] - q(s,a)]
                self.Q[curr_state][curr_action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[curr_state][curr_action])
                self.update_policy_average(curr_state)                  # 更新行为策略
                curr_state = next_state
            # if episode % 20 == 0:
            #     print(self.Q[0])
        return wrong_count, self.Q[0], self.Q[1]

    # 两个相同价值的动作只选择第一个
    def update_policy_max(self, state): # 按e-greedy策略更新行为策略
        best_action = np.argmax(self.Q[state])
        self.behavior_policy[state][:] = self.epsilon/(self.nA[state]-1)
        self.behavior_policy[state][best_action] = 1 - self.epsilon

    # 两个相同价值的动作具有相同的概率被选择，而不是只选择第一个
    def update_policy_average(self, state):
        best_actions = np.argwhere(self.Q[state] == np.max(self.Q[state]))
        best_actions_count = len(best_actions)
        if best_actions_count == self.nA[state]:
            self.behavior_policy[state][:] = 1 / best_actions_count
        else:
            for action in range(self.nA[state]):
                if action in best_actions:
                    self.behavior_policy[state][action] = (1 - self.epsilon) / best_actions_count
                else:
                    self.behavior_policy[state][action] = self.epsilon / (self.nA[state] - best_actions_count)

    def choose_action(self, state): # 选择动作，从行为策略中按概率选择
        action = np.random.choice(self.nA[state], p=self.behavior_policy[state])
        return action


class TD_DQLearning(TD_QLearning):
    def __init__(self, env, episodes, policy, alpha, gamma, epsilon):
        super().__init__(env, episodes, policy, alpha, gamma, epsilon)
        self.nA = [2,10]
        self.Q1 = {
            0: np.zeros(2),
            1: np.zeros(10),
            2: np.zeros(2),
        }
        self.Q2 = {
            0: np.zeros(2),
            1: np.zeros(10),
            2: np.zeros(2),
        }

    def update_policy_average(self, state):
        Q = (self.Q1[state] + self.Q2[state])/2
        best_actions = np.argwhere(Q == np.max(Q))
        best_actions_count = len(best_actions)
        if best_actions_count == self.nA[state]:
            self.behavior_policy[state][:] = 1 / best_actions_count
        else:
            for action in range(self.nA[state]):
                if action in best_actions:
                    self.behavior_policy[state][action] = (1 - self.epsilon) / best_actions_count
                else:
                    self.behavior_policy[state][action] = self.epsilon / (self.nA[state] - best_actions_count)

    def run(self):
        id = 1  # 第一次用Q1
        wrong_count = np.zeros(self.episodes)
        for episode in range(self.episodes):                    # 分幕
            curr_state, _ = self.env.reset()
            done = False
            while not done:  # 幕内采样
                curr_action = self.choose_action(curr_state)    # 选择动作
                if curr_state == 0 and curr_action == 1: # right to S_M
                    wrong_count[episode] = 1
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                if id == 1:
                    id = 2  # 下一次用Q2
                    best_action = np.argmax(self.Q1[next_state])
                    self.Q1[curr_state][curr_action] += self.alpha * (reward + self.gamma * self.Q2[next_state][best_action] - self.Q1[curr_state][curr_action])
                else:  # assert id == 2
                    id = 1  # 下一次用Q1
                    best_action = np.argmax(self.Q2[next_state])
                    self.Q2[curr_state][curr_action] += self.alpha * (reward + self.gamma * self.Q1[next_state][best_action] - self.Q2[curr_state][curr_action])
                self.update_policy_average(curr_state)                  # 更新行为策略
                curr_state = next_state
            # if episode % 20 == 0:
            #     print((self.Q1[0] + self.Q2[0])/2)
        return wrong_count, (self.Q1[0] + self.Q2[0])/2, (self.Q1[1] + self.Q2[1])/2


if __name__ == "__main__":
    env = Env()
    Episodes = 300  # 300
    behavior_policy = {
        0: np.ones((2))/2,
        1: np.ones((10))/10,
    }
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    runs = 1000  # 1000

    wrong_count_q = np.zeros((runs, Episodes))
    wrong_count_dq = np.zeros((runs, Episodes))
    Qs = np.zeros(2)
    DQs = np.zeros(2)
    Qm = np.zeros(10)
    DQm = np.zeros(10)
    for run in tqdm.trange(runs):
        ctrl = TD_QLearning(env, Episodes, behavior_policy, alpha, gamma, epsilon)
        wrong_count, Qs, Qm = ctrl.run()
        Qs += Qs
        Qm += Qm
        wrong_count_q[run] = wrong_count
        ctrl = TD_DQLearning(env, Episodes, behavior_policy, alpha, gamma, epsilon)
        wrong_count, DQs, DQm = ctrl.run()
        DQs += DQs
        DQm += DQm
        wrong_count_dq[run] = wrong_count
    q = np.mean(wrong_count_q, axis=0)
    dq = np.mean(wrong_count_dq, axis=0)
    plt.plot(q, label="Q-Learning", linestyle='-')
    plt.plot(dq, label="D-Q-Learning", linestyle=':')
    plt.plot(np.ones(Episodes) * 0.05, label='最优参考值', linestyle='--')
    plt.legend()
    plt.xlabel("幕")
    plt.ylabel("错误动作比例")
    plt.grid()
    plt.show()
    helper.print_seperator_line(helper.SeperatorLines.middle, "Q-Learning - Q(S)")
    print(np.round(Qs/runs, 4))
    helper.print_seperator_line(helper.SeperatorLines.middle, "Q-Learning - Q(M)")
    print(np.round(Qm/runs, 4))
    helper.print_seperator_line(helper.SeperatorLines.middle, "Double Q-Learning - Q(S)")
    print(np.round(DQs/runs, 4))
    helper.print_seperator_line(helper.SeperatorLines.middle, "Double Q-Learning - Q(M)")
    print(np.round(DQm/runs, 4))
