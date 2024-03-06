import numpy as np
import matplotlib.pyplot as plt

LEFT, DOWN, RIGHT, UP  = 0, 1, 2, 3

class GridWorld(object):
    # 生成环境
    def __init__(self, 
        GridWidth: int, 
        GridHeight: int, 
        StartStates: list, 
        EndStates: list, 
        Actions: list,
        Policy: list, 
        Transition: list, 
        StepReward: float,
        SpecialReward: dict, 
        SpecialMove: dict, 
        Blocks: list):

        self.Width = GridWidth
        self.Height = GridHeight
        self.Actions = Actions
        self.nS = GridHeight * GridWidth
        self.nA = len(Actions)
        self.SpecialReward = SpecialReward
        self.StartStates = StartStates
        self.EndStates = EndStates
        self.SpecialMove = SpecialMove
        self.Blocks = Blocks
        self.Policy = self.__init_policy(Policy)
        self.P_S_R = self.__init_states(Transition, StepReward)

    # 把统一的policy设置复制到每个状态上
    def __init_policy(self, Policy):
        if Policy is None or len(Policy) == 0:
            Policy = [1/self.nA for _ in range(self.nA)]
        if isinstance(Policy, list):
            self_policy = {}
            for s in range(self.nS):
                self_policy[s] = Policy
            return self_policy
        elif isinstance(Policy, dict):  # {0:[...], 1:[...], ...}
            return Policy
        else:
            raise Exception("Policy 类型错误")

    # 用于生成状态->动作->转移->奖励字典
    def __init_states(self, Transition, StepReward):
                #       s0 a0  p  s' r    p  s' r       a1             s1
        P = {}  # dict {0:{0:[(p, s, r), (p, s, r)...], 1:[...], ...}, 1:{...}, ...}
        # 建立 0-based ID 与方格世界中(x,y)的对应关系
        # 比如，0-(0,0), 1-(0,1)...
        s_id = 0
        self.Pos2Sid = {}
        self.Sid2Pos = {}
        for y in range(self.Height):
            for x in range(self.Width):
                self.Pos2Sid[x,y] = s_id
                self.Sid2Pos[s_id] = [x,y]
                s_id += 1
        # 遍历每个状态极其位置
        for s, (x,y) in self.Sid2Pos.items():
            P[s] = {}
            if (s in self.EndStates):
                continue
            for action in self.Actions:
                list_probs = []
                for dir, prob in enumerate(Transition):
                    if (prob == 0.0):
                        continue
                    s_next = self.__get_next_state(
                        s, x, y, action + dir - 1)   # 处理每一个转移概率，方向逆时针减1
                    reward = StepReward              # 通用奖励定义 (-1)
                    if (s, s_next) in self.SpecialReward:    # 如果有特殊奖励定义
                        reward = self.SpecialReward[(s, s_next)]
                    list_probs.append((prob, s_next, reward))
                # 为状态|动作组合设置转移概率
                P[s][action] = list_probs
        return P

    # 用于计算移动后的下一个状态
    # 左上角为 [0,0], 横向为 x, 纵向为 y
    def __get_next_state(self, s, x, y, action):
        action = action % 4         # 避免负数
        if (s,action) in self.SpecialMove:
            return self.SpecialMove[(s,action)]

        if (action == UP):          # 向上转移
            if (y != 0):            # 不在上方边界处，否则停在原地不动
                s = s - self.Width
        elif (action == DOWN):      # 向下转移
            if (y != self.Height-1):# 不在下方边界处，否则停在原地不动
                s = s + self.Width
        elif (action == LEFT):      # 向左转移
            if (x != 0):            # 不在左侧边界处，否则停在原地不动
                s = s - 1
        elif (action == RIGHT):     # 向右转移
            if (x != self.Width-1): # 不在右侧边界处，否则停在原地不动
                s = s + 1
        return s

    def is_end(self, s):
        return (s in self.EndStates)

    def get_actions(self, s):
        actions = self.P_S_R[s]
        return actions.items()

action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']

def print_P(P):
    print("状态->动作->转移->奖励 字典：")
    for s,v in P.items():
        print("state =",s)
        for action,v2 in v.items():
            print("\taction =", action_names[action])
            print("\t",v2)

