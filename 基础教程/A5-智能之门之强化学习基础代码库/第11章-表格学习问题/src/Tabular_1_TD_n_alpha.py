import common.Algo_DP_PolicyEvaluation as algoDP
import numpy as np
import common.CommonHelper as helper
import gymnasium as gym
import common.Algo_TD_TDn as algoTDn
import common.Algo_TD_TD0 as algoTD0
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
import multiprocessing as mp


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

def single_run_TD0(env, episodes, behavior_policy, alpha, gamma, n):
    pred = algoTD0.TD_TD0(env, episodes, behavior_policy, alpha=alpha, gamma=gamma)
    V = pred.run()
    return V

if __name__=="__main__":
    behavior_policy = np.ones((N_STATES, 2)) / 2
    gamma = 1

    helper.print_seperator_line(helper.SeperatorLines.long, "DP 策略评估")
    env = Env()
    V_dp, _ = algoDP.calculate_VQ_pi(env, behavior_policy, gamma)    # 迭代计算V,Q
    helper.print_V(V_dp, 1, (1,21), helper.SeperatorLines.middle, "V 值")

    episodes = 10
    runs = 100
    n_steps = [1, 2, 4, 8, 16, 32, 64, 128]
    markers = ['.', 'o', 'v', 's', 'x', '>', 'p', '*']
    alphas = np.arange(0., 1.0, 0.1)
    errors = np.zeros((len(n_steps), len(alphas)))
    # for run in range(runs):
    #     for n_ind, n in enumerate(n_steps):
    #         for alpha_ind, alpha in enumerate(alphas):
    #             print('n = %d, alpha = %.2f' % (n, alpha))
    #             pool = mp.Pool(processes=4)
    #             results = []
    #             for _ in range(episodes):
    #                 results.append(pool.apply_async(single_run_TDn, args=(env, episodes, behavior_policy, alpha, gamma, n)))
    #             pool.close()
    #             pool.join()

    #             for i in range(episodes):
    #                 V = results[i].get()
    #                 # calculate the RMS error
    #                 errors[n_ind, alpha_ind] += np.sqrt(np.sum(np.power(V - V_dp, 2)) / (N_STATES))
    # # take average
    # errors /= runs * episodes

    for run in range(runs):
        for n_ind, n in enumerate(n_steps):
            pool = mp.Pool(processes=4)
            results = []
            for alpha_ind, alpha in enumerate(alphas):
                print('run = %d, n = %d, alpha = %.1f' % (run, n, alpha))
                results.append(pool.apply_async(single_run_TDn, args=(env, episodes, behavior_policy, alpha, gamma, n)))
            pool.close()
            pool.join()
            for alpha_ind, alpha in enumerate(alphas):
                V = results[alpha_ind].get()
                # calculate the RMS error
                errors[n_ind, alpha_ind] += np.sqrt(np.sum(np.power(V - V_dp, 2)) / (N_STATES))
    errors /= runs

    np.save('errors.npy', errors)

    for i in range(0, len(n_steps)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (n_steps[i]), marker=markers[i])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'误差')
    plt.legend()
    plt.grid()
    plt.show()


