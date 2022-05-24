import numpy as np
from enum import Enum

# 状态
class States(Enum):
    A = 0
    B = 1
    Home = 2
    Hill = 3

# Reward
Home = 1
Hill = 0

# 状态奖励
Rewards = [0, 0, Home, Hill]

TransMatrix = np.array(
    [   #A     B   T1   T2
        [0, 1, 0, 0], 
        [0, 0, 3/4, 1/4],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
)

class A_B_Question(object):
    def __init__(self):
        self.num_states = len(States)

    def get_states(self) -> States:
        return States
        
    def get_states_count(self) -> int:
        return len(States)

    def random_select_state(self):
        state_value = np.random.choice(self.num_states)
        return States(state_value)

    def step(self, curr_state: States):
        next_state_value = np.random.choice(self.num_states, p=TransMatrix[curr_state.value])
        reward = Rewards[next_state_value]
        return States(next_state_value), reward

    def step2(self, curr_state: States):
        if self.is_end_state(curr_state):
            reward = Rewards[curr_state.value]
            return curr_state, reward, True
        else:
            next_state_value = np.random.choice(self.num_states, p=TransMatrix[curr_state.value])
            reward = Rewards[next_state_value]
            return States(next_state_value), reward, False


    def get_reward(self, curr_state: States):
        return Rewards[curr_state.value]

    def is_end_state(self, curr_state: States):
        if (curr_state in [States.Home, States.Hill]):
            return True
        else:
            return False