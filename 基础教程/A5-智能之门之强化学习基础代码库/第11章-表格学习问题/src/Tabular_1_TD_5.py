from enum import Enum
import common.Algo_DP_PolicyEvaluation as algoDP
import numpy as np
import common.CommonHelper as helper
import gymnasium as gym
import common.Algo_TD_TDn as algoTDn
import common.Algo_TD_TD0 as algoTD0
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False


N_STATES = 21
START_STATE = 10
ACTION_LEFT = 0
ACTION_RIGHT = 1
STEP_REWARD = 0
END_STATES = [0, 20]

class Env(object):
    def __init__(self, policy=None):
        self.observation_space = gym.spaces.Discrete(N_STATES)
        self.action_space = gym.spaces.Discrete(2)
        self.Policy = policy
        self.end_states = END_STATES
        self.P = self.initialize_P()

    def initialize_P(self):
        P = {}
        for s in range(N_STATES):
            P[s] = {}
            P[s][ACTION_LEFT] = []
            P[s][ACTION_RIGHT] = []
            if s == 0 or s == 20:
                P[s][ACTION_LEFT].append((1.0, s, 0, True))
                P[s][ACTION_RIGHT].append((1.0, s, 0, True))
            else:
                if s == 1:
                    P[s][ACTION_LEFT].append((1.0, 0, -1, True))
                    P[s][ACTION_RIGHT].append((1.0, s+1, 0, True))
                elif s == 19:
                    P[s][ACTION_RIGHT].append((1.0, 20, +1, True))
                    P[s][ACTION_LEFT].append((1.0, s-1, 0, True))
                else:
                    P[s][ACTION_LEFT].append((1.0, s - 1, STEP_REWARD, False))
                    P[s][ACTION_RIGHT].append((1.0, s + 1, STEP_REWARD, False))
        return P

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.state = START_STATE
        return START_STATE, 0

    def step(self, a):
        if a == ACTION_LEFT:
            s_next = self.state - 1
        else:
            s_next = self.state + 1
        if s_next == 0:
            r = -1
        elif s_next == 20:
            r = 1
        else:
            r = STEP_REWARD
        self.state = s_next
        is_end = s_next in self.end_states
        return s_next, r, is_end, None, None

    def is_end(self,s):
        if s in self.end_states:
            return True
        else:
            return False


def single_run_TDn(env, Episodes, behavior_policy, alpha, gamma, n):
    pred = algoTDn.TD_n(env, Episodes, behavior_policy, alpha=alpha, gamma=gamma, n=n)
    V = pred.run()
    return V

def single_run_TD0(env, Episodes, behavior_policy, alpha, gamma, n):
    pred = algoTD0.TD_TD0(env, episode, behavior_policy, alpha=alpha, gamma=gamma)
    V = pred.run()
    return V

if __name__=="__main__":
    behavior_policy = np.ones((N_STATES, 2)) / 2
    gamma = 1
    n = 5
    alpha = 0.1

    helper.print_seperator_line(helper.SeperatorLines.long, "DP 策略评估")
    env = Env()
    V_dp, _ = algoDP.calculate_VQ_pi(env, behavior_policy, gamma)    # 迭代计算V,Q
    helper.print_V(V_dp, 1, (1,21), helper.SeperatorLines.middle, "V 值")
    plt.plot(V_dp, label='基准')
    
    episodes = [10, 50, 100]
    lines = ["-", "--", ":", "-."]
    for episode in episodes:
        V = single_run_TDn(env, episode, behavior_policy, alpha, gamma, n)
        print(helper.Norm2Err(V, V_dp))
        plt.plot(V, linestyle=lines[episodes.index(episode)], label=r'%d 幕' % (episode))
    
    plt.legend()
    plt.grid()
    plt.xlabel('状态')
    plt.ylabel('V 值')
    plt.show()
