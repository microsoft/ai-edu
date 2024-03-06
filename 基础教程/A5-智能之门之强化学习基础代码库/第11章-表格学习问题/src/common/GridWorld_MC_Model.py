import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

LEFT, DOWN, RIGHT, UP  = 3, 2, 1, 0

class GridWorld(object):
    # 生成环境
    def __init__(self, 
        GridWidth: int, 
        GridHeight: int, 
        StartStates: list, 
        EndStates: list, 
        Actions: list,
        Transition: list,
        GoalReward: float,
        StepReward: float,
        SpecialReward: dict, 
        SpecialMove: dict, 
        Blocks: list):

        self.Width = GridWidth
        self.Height = GridHeight
        self.Actions = Actions
        self.observation_space = gym.spaces.Discrete(GridHeight * GridWidth)
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.nA = self.observation_space.n
        self.nS = self.action_space.n
        self.GoalReward = GoalReward
        self.StepReward = StepReward
        self.SpecialReward = SpecialReward
        self.StartStates = StartStates
        self.EndStates = EndStates
        self.SpecialMove = SpecialMove
        self.Blocks = Blocks
        self.__init_states()

    def reset(self):
        self.state = self.StartStates[0]
        return self.state, None

    # 用于生成状态->动作->转移->奖励字典
    def __init_states(self):
                #       s0 a0  p  s' r    p  s' r       a1             s1
        P = {}  # dict {0:{0:[(p, s, r), (p, s, r)...], 1:[...], ...}, 1:{...}, ...}
        # 建立 0-based ID 与方格世界中(x,y)的对应关系
        # 比如，0:(0,0), 1:(0,1)...
        s_id = 0
        self.Pos2Sid = {}
        self.Sid2Pos = {}
        for y in range(self.Height):
            for x in range(self.Width):
                self.Pos2Sid[x,y] = s_id
                self.Sid2Pos[s_id] = [x,y]
                s_id += 1


    # 用于计算移动后的下一个状态
    # 左上角为 [0,0], 横向为 x, 纵向为 y
    def __get_next_state(self, s, x, y, action):
        action = action % 4         # 避免负数
        if (s,action) in self.SpecialMove:
            return self.SpecialMove[(s,action)]

        s_next = s
        if action == UP:          # 向上转移
            if y != 0:            # 不在上方边界处，否则停在原地不动
                s_next = s - self.Width
        elif action == DOWN:      # 向下转移
            if y != self.Height-1:# 不在下方边界处，否则停在原地不动
                s_next = s + self.Width
        elif action == LEFT:      # 向左转移
            if x != 0:            # 不在左侧边界处，否则停在原地不动
                s_next = s - 1
        elif action == RIGHT:     # 向右转移
            if x != self.Width-1: # 不在右侧边界处，否则停在原地不动
                s_next = s + 1
        if s_next in self.Blocks:
            s_next = s  # 保留在原地不动
        return s_next

    def is_end(self, s):
        return s in self.EndStates

    def get_actions(self, s):
        actions = self.P_S_R[s]
        return actions.items()
    
    def step(self, action):
        x, y = self.Sid2Pos[self.state]
        self.state = self.__get_next_state(self.state, x, y, action)
        if self.state in self.EndStates:
            reward = self.GoalReward
        elif self.state in self.SpecialReward:
            reward = self.SpecialReward[self.state]
        else:
            reward = self.StepReward
        return self.state, reward, self.state in self.EndStates, None, None


