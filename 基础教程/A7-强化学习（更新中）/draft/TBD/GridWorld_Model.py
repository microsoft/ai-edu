import numpy as np
import copy
import matplotlib.pyplot as plt

LEFT, UP, RIGHT, DOWN  = 0, 1, 2, 3

class GridWorld(object):
    # 生成环境
    def __init__(self, 
        GridWidth, GridHeight, StartStates, EndStates, 
        Actions, Policy, SlipProbs, 
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
        self.P_S_R = self.__init_states(SlipProbs, StepReward)

    # 把统一的policy设置复制到每个状态上
    def __init_policy(self, Policy):
        PI = {}
        for s in range(self.nS):
            PI[s] = Policy
        return PI

    # 用于生成状态->动作->转移->奖励字典
    def __init_states(self, Probs, StepReward):
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
                for dir, prob in enumerate(Probs):
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

'''
# 式 (2.1) 计算 q_pi
def q_pi(Psr, gamma, V):
    q = 0
    # 遍历每个转移概率,以计算 q_pi
    for p, s_next, r in Psr:
        # math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
        q += p * (r + gamma * V[s_next])
    return q

# 式 (5) 计算 v_pi
def v_pi(env: GridWorld, s, gamma, V, Q):
    actions = env.get_actions(s)    # 获得当前状态s下的所有可选动作
    v = 0
    for a, p_s_r in actions:        # 遍历每个动作以计算q值，进而计算v值
        q = q_pi(p_s_r, gamma, V)
        # math: \sum_a \pi(a|s) q_\pi (s,a)
        v += env.Policy[a] * q
        # 顺便记录下q(s,a)值,不需要再单独计算一次
        Q[s,a] = q
    return v

# 迭代法计算 v_pi
def V_in_place_update(env: GridWorld, gamma, iteration):
    V = np.zeros(env.nS)            # 初始化 V(s)
    Q = np.zeros((env.nS, env.nA))  # 初始化 Q(s,a)
    count = 0   # 计数器，用于衡量性能和避免无限循环
    # 迭代
    while (count < iteration):
        V_old = V.copy()    # 保存上一次的值以便检查收敛性
        # 遍历所有状态 s
        for s in range(env.nS):
            V[s] = v_pi(env, s, gamma, V, Q)
        # 检查收敛性
        if abs(V-V_old).max() < 1e-4:
            break
        count += 1
    # end while
    print("迭代次数 = ",count)
    return V, Q

# 双数组迭代
def V_pi_2array(env: GridWorld, gamma, iteration):
    V = np.zeros(env.nS)
    Q = np.zeros((env.nS, env.nA))

    count = 0
    # 迭代
    while (count < iteration):
        V_old = V.copy()
        # 遍历所有状态 s
        for s in range(env.nS):
            v_pi = 0
            # 获得 状态->动作 策略概率
            actions = env.get_actions(s)
            if actions is not None:
                # 遍历每个策略概率

                for action, next_p_s_r in actions:
                    # 获得 动作->状态 转移概率
                    q_pi = 0
                    # 遍历每个转移概率,以计算 q_pi
                    for p, s_next, r in next_p_s_r:
                        # 式2.1 math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
                        q_pi += p * (r + gamma * V_old[s_next])
                    #end for
                    # 式5 math: \sum_a \pi(a|s) q_\pi (s,a)
                    Q[s][action] = q_pi
                    v_pi += 0.25 * q_pi
                # end for
            V[s] = v_pi
        #endfor
        # 检查收敛性
        if abs(V-V_old).max() < 1e-4:
            break
        # 把 V_curr 赋值给 V_next
        #print(np.reshape(np.round(V,2), (4,4)))
        count += 1
    # end while
    print(count)
    #print(Q)
    return V, Q


def V_star(env: GridWorld, gamma, iteration):
    V_star = np.zeros(env.nS)
    Q_star = copy.deepcopy(env.Psr)    # 拷贝状态转移结构以存储Q(s,a)值
    count = 0
    # 迭代
    while (count < iteration):
        V_old = V_star.copy()
        # 遍历所有状态 s
        for s in range(env.nS):
            # 获得 状态->动作 策略概率
            actions = env.get_actions(s)
            list_q = []
            if actions is not None:
                # 遍历每个策略概率
                for action, next_p_s_r in actions:
                    # 获得 动作->状态 转移概率
                    q_star = 0
                    # 遍历每个转移概率,以计算 q_pi
                    for p, s_next, r in next_p_s_r:
                        # 式2.1 math: \sum_{s'} p_{ss'}^a [ r_{ss'}^a + \gamma *  v_{\pi}(s')]
                        q_star += p * (r + gamma * V_star[s_next])
                    #end for
                    # 式5 math: \sum_a \pi(a|s) q_\pi (s,a)
                    list_q.append(q_star)
                    Q_star[s][action] = q_star
                # end for
            V_star[s] = max(list_q) if len(list_q) > 0 else 0
        #endfor
        # 检查收敛性
        if abs(V_star-V_old).max() < 1e-4:
            break
        count += 1
    # end while
    print(count)
    return V_star, Q_star

def get_policy(env: GridWorld, V, gamma):
    policy = np.zeros((env.nS, env.nA))    
    for s in range(env.nS):
        actions = env.get_actions(s)
        list_q = np.zeros(env.nA)
        if actions is None:
            continue
        # 遍历每个策略概率
        for action, next_p_s_r in actions:
            q_star = 0
            for p, s_next, r in next_p_s_r:
                q_star += p * (r + gamma * V[s_next])
            list_q[action] = q_star

        policy[s, np.argmax(list_q)] = 1
    return policy
'''
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

