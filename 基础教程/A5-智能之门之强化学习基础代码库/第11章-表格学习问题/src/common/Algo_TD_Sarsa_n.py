
from common.Algo_TD_n_Base import TD_n_Base
import tqdm
import numpy as np

class SARSA_n(TD_n_Base):
    def run(self):
        rewards = []
        for episode in tqdm.trange(self.episodes):
            episode_reward = 0
            s, _ = self.env.reset()
            a = np.random.choice(self.env.action_space.n, p=self.behavior_policy[s])
            reward_Trajactor = [0]
            sa_Trajectory = [(s,a)]
            T = np.inf
            t = 0
            while True:
                if t < T:
                    s_next, reward, done, truncated, _ = self.env.step(a)
                    reward_Trajactor.append(reward)
                    episode_reward += reward
                    if done:
                        T = t + 1
                        sa_Trajectory.append((s_next, a))
                    else:
                        a_next = np.random.choice(self.env.action_space.n, p=self.behavior_policy[s_next])
                        sa_Trajectory.append((s_next, a_next))
                # end if
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += pow(self.gamma, i-tau-1) * reward_Trajactor[i]
                    
                    if tau + self.n < T:
                        G += pow(self.gamma, self.n) * self.Q[sa_Trajectory[tau + self.n]]
                    self.Q[sa_Trajectory[tau]] += self.alpha * (G - self.Q[sa_Trajectory[tau]])
                    self.update_policy_average(s)

                if tau == T - 1:
                    rewards.append(episode_reward)
                    break            
                s = s_next
                a = a_next
                t += 1
        return self.Q, rewards
