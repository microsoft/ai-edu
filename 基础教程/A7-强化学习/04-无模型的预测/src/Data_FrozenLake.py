import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Start = 0
    Safe1 = 1
    Hole2 = 2
    Safe3 = 3
    Safe4 = 4
    Safe5 = 5
    Safe6 = 6
    Safe7 = 7
    Hole8 = 8
    Safe9 = 9
    Hole10 = 10
    Safe11 = 11
    Safe12 = 12
    Safe13 = 13
    Safe14 = 14
    Goal15 = 15

# Reward
Hole = -1
Goal = 5

# 状态奖励
Rewards = [0, 0, Hole, 0,   
           0, 0, 0, 0,   
          Hole, 0, Hole, 0,
           0, 0, 0, Goal]

TransMatrix = np.array(
    [   
        [0.0, 1/2, 0.0, 0.0, 
         1/2, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 0

        [1/3, 0.0, 1/3, 0.0, 
         0.0, 1/3, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 1

        [0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 2, Hole

        [0.0, 0.0, 1/2, 0.0, 
         0.0, 0.0, 0.0, 1/2, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 3

        [1/3, 0.0, 0.0, 0.0, 
         0.0, 1/3, 0.0, 0.0, 
         1/3, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 4

        [0.0, 1/4, 0.0, 0.0, 
         1/4, 0.0, 1/4, 0.0, 
         0.0, 1/4, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 5

        [0.0, 0.0, 1/4, 0.0, 
         0.0, 1/4, 0.0, 1/4, 
         0.0, 0.0, 1/4, 0.0,
         0.0, 0.0, 0.0, 0.0],  # 6

        [0.0, 0.0, 0.0, 1/3, 
         0.0, 0.0, 1/3, 0.0, 
         0.0, 0.0, 0.0, 1/3, 
         0.0, 0.0, 0.0, 0.0],  # 7

        [0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 8, Hole

        [0.0, 0.0, 0.0, 0.0, 
         0.0, 1/4, 0.0, 0.0, 
         1/4, 0.0, 1/4, 0.0, 
         0.0, 1/4, 0.0, 0.0],  # 9

        [0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 10, Hole

        [0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 1/3, 
         0.0, 0.0, 1/3, 0.0, 
         0.0, 0.0, 0.0, 1/3],  # 11

        [0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         1/2, 0.0, 0.0, 0.0, 
         0.0, 1/2, 0.0, 0.0],  # 12

        [0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 1/3, 0.0, 0.0, 
         1/3, 0.0, 1/3, 0.0],  # 13

        [0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 1/3, 0.0, 
         0.0, 1/3, 0.0, 1/3],  # 14

        [0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0],  # 15, end state, no transform
    ]
)

class Data_Frozen_Lake(object):
    def __init__(self):
        self.num_states = len(States)

    def get_TransMatrix(self):
        return TransMatrix

    def get_states(self) -> States:
        return States
        
    def get_states_count(self) -> int:
        return len(States)

    def get_start_state(self):
        return States.Start

    def random_select_state(self):
        state_value = np.random.choice(16)
        return States(state_value)

    def step(self, curr_state: States):
        next_state_value = np.random.choice(self.num_states, p=TransMatrix[curr_state.value])
        reward = Rewards[next_state_value]
        next_state = States(next_state_value)
        return next_state, reward, self.is_end_state(next_state)

    def step2(self, curr_state: States):
        if self.is_end_state(curr_state):
            reward = Rewards[curr_state.value]
            return curr_state, reward, True
        else:
            next_state_value = np.random.choice(self.num_states, p=TransMatrix[curr_state.value])
            reward = Rewards[next_state_value]
            return States(next_state_value), reward, False

    def get_rewards(self):
        return Rewards

    def get_reward(self, curr_state: States):
        return Rewards[curr_state.value]

    def is_end_state(self, curr_state: States):
        if (curr_state in [States.Hole2, States.Hole10, States.Hole8, States.Goal15]):
            return True
        else:
            return False