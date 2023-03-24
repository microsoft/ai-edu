import numpy as np
import matplotlib.pyplot as plt

LEFT, UP, RIGHT, DOWN  = 0, 1, 2, 3

class GridWorld(object):
    # 生成环境
    def __init__(self, 
        GridWidth, GridHeight, StartStates, EndStates, 
        Actions, Policy, Transition, 
        StepReward, SpecialReward, 
        SpecialMove, Blocks):

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
        PI = {}
        for s in range(self.nS):
            PI[s] = Policy
        return PI

    # 用于生成状态->动作->转移->奖励字典
    def __init_states(self, Transition, StepReward):
        P = {}
        s_id = 0
        self.Pos2Sid = {}
        self.Sid2Pos = {}
        for y in range(self.Height):
            for x in range(self.Width):
                self.Pos2Sid[x,y] = s_id
                self.Sid2Pos[s_id] = [x,y]
                s_id += 1

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
                        s, x, y, action + dir - 1)    # 处理每一个转移概率，方向逆时针减1
                    reward = StepReward              # 通用奖励定义 (-1)
                    if (s, s_next) in self.SpecialReward:    # 如果有特殊奖励定义
                        reward = self.SpecialReward[(s, s_next)]
                    list_probs.append((prob, s_next, reward))
                
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

action_names = ['LEFT', 'UP', 'RIGHT', 'DOWN']

def print_P(P):
    print("状态->动作->转移->奖励 字典：")
    for s,v in P.items():
        print("state =",s)
        for action,v2 in v.items():
            print("\taction =", action_names[action])
            print("\t",v2)

        # left,  up,     right,  down
chars = [0x2190, 0x2191, 0x2192, 0x2193]
# 需要处理多个值相等的情况
def print_policy(policy, shape):
    best_actions = np.argmax(policy, axis=1)
    for i, action in enumerate(best_actions):
        print(chr(chars[action]), end="")
        print(" ", end="")
        if ((i+1) % shape[0] == 0):
            print("")

# 绘图
def draw_table(V, shape):
    tab = plt.table(cellText=V, loc='center', rowHeights=[0.1]*5)
    tab.scale(1,1)
    plt.axis('off')
    plt.show()

