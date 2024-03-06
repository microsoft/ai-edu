import numpy as np
import matplotlib.pyplot as plt
from common.Algo_TD_Sarsa_n import SARSA_n
import common.CommonHelper as helper
import common.DrawQpi as drawQpi
import gymnasium as gym

# world height width
WORLD_HEIGHT, WORLD_WIDTH = 7, 10

#   x -->
# y (0,0) (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0) (9,0)
# | (0,1) (1,1) (2,1) (3,1) (4,1) (5,1) (6,1) (7,1) (8,1) (9,1)
# | ...
# | (0,6) (1,6) (2,6) (3,6) (4,6) (5,6) (6,6) (7,6) (8,6) (9,6)

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# possible actions
ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT = 0, 1, 2, 3
action_name = ["上", "右", "下", "左"]
# reward for each step
STEP_REWARD = 0
GOAL_REWARD = 1
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

def converter_action_to_policy(action):
    if action == ACTION_UP:
        return np.array([1, 0, 0, 0])
    elif action == ACTION_RIGHT:
        return np.array([0, 1, 0, 0])
    elif action == ACTION_DOWN:
        return np.array([0, 0, 1, 0])
    elif action == ACTION_LEFT:
        return np.array([0, 0, 0, 1])
    
def manual_set_behavior_policy(b_policy):
    action_right = [30, 31, 32, 33, 24, 15, 6, 7, 8]
    action_down = [9, 19, 29, 39]
    action_left = [49, 48]
    for s in action_left:
        b_policy[s] = converter_action_to_policy(ACTION_LEFT)
    for s in action_down:
        b_policy[s] = converter_action_to_policy(ACTION_DOWN)
    for s in action_right:
        b_policy[s] = converter_action_to_policy(ACTION_RIGHT)

if __name__ == "__main__":
    epsilon = 0.1
    alpha = 0.1
    gamma = 1.0
    episodes = 1
    env = WindyGridWorld()
    env.state = 36
    info = env.step(ACTION_RIGHT)
    print(info)
    env.state = 38
    info = env.step(ACTION_LEFT)
    print(info)

    for n in [1, 5, 10]:
        helper.print_seperator_line(helper.SeperatorLines.middle, f"SARSA({n})")
        s, _ = env.reset()
        behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
        manual_set_behavior_policy(behavior_policy)
        ctrl = SARSA_n(env, episodes, behavior_policy, alpha=alpha, gamma=gamma, n=n, epsilon=epsilon)
        Q, rewards = ctrl.run()
        for i in range(Q.shape[0]):
            if Q[i].sum() != 0:
                print(f"state {i}: {Q[i]}")
        drawQpi.drawQ(Q, (7,10), round=8, goal_state=37)

