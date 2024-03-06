import numpy as np
import matplotlib.pyplot as plt
from common.Algo_TD_Sarsa_n import SARSA_n
import common.CommonHelper as helper
import common.DrawQpi as drawQpi
import gymnasium as gym
import scipy.signal as ss
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False

# world height width
WORLD_HEIGHT, WORLD_WIDTH = 7, 10
# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# possible actions
ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT = 0, 1, 2, 3
action_name = ["上", "右", "下", "左"]
# reward for each step
STEP_REWARD = -1
GOAL_REWARD = 0
START = 30
GOAL = 37
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

class WindyGridWorld(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Discrete(WORLD_HEIGHT * WORLD_WIDTH)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

    def reset(self):
        self.state = START
        return self.state, None

    def state_to_xy(self, state):
        return state % WORLD_WIDTH, state // WORLD_WIDTH

    def xy_to_state(self, x, y):
        return y * WORLD_WIDTH + x

    def step(self, action):
        i, j = self.state_to_xy(self.state)
        if action == ACTION_UP:
            x = i
            y = j - 1 - WIND[i]
        elif action == ACTION_DOWN:
            x = i
            y = j + 1 - WIND[i]
        elif action == ACTION_LEFT:
            x = max(i - 1, 0)
            y = j - WIND[i]
        elif action == ACTION_RIGHT:
            x = min(i + 1, WORLD_WIDTH - 1)
            y = j - WIND[i]
        else:
            assert False
        if y < 0:
            y = 0
        if y > WORLD_HEIGHT - 1:
            y = WORLD_HEIGHT - 1
        self.state = self.xy_to_state(x, y)
        reward = GOAL_REWARD if self.state == GOAL else STEP_REWARD
        return self.state, reward, self.state == GOAL, None, None


if __name__ == "__main__":
    epsilon = 0.1
    alpha = 0.1
    gamma = 1.0
    episodes = 500
    env = WindyGridWorld()

    lables = ["SARSA(0)", "SARSA(5)", "SARSA(10)"]
    lines = ["-", "--", "-."]
    ns = [1, 5, 10]
    for j in range(len(ns)):
        n = ns[j]
        helper.print_seperator_line(helper.SeperatorLines.middle, f"SARSA({n})")
        s, _ = env.reset()
        behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        ctrl = SARSA_n(env, episodes, behavior_policy, alpha=alpha, gamma=gamma, n=n, epsilon=epsilon)
        Q, R = ctrl.run()

        helper.print_Q(Q, 2, (WORLD_HEIGHT, WORLD_WIDTH), helper.SeperatorLines.middle, "Q 值")
        drawQpi.drawQ(Q, (WORLD_HEIGHT, WORLD_WIDTH), round=2, goal_state=37)

        # play
        state, _ = env.reset()
        i = 1
        while True:
            action = np.argmax(Q[state])
            next_state, reward, done, _, _ = env.step(action)
            print(f"step: {i}, state: {state}, action: {action_name[action]}, reward: {reward}, next_state: {next_state}")
            state = next_state
            i += 1
            if done or i > 100:
                break

        tmp = ss.savgol_filter(R[5:], 40, 2)
        plt.plot(tmp, label=lables[j], linestyle=lines[j])
    plt.legend()
    plt.grid()
    plt.xlabel("幕")
    plt.ylabel("每幕的回报")
    plt.show()
