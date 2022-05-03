import numpy as np
from enum import Enum
import copy

LEFT, UP, RIGHT, DOWN  = 0, 1, 2, 3

class GridWorld(object):
    # 生成环境
    def __init__(self, GridWidth, GridHeight, Actions, SpecialReward, Prob, StepReward, EndStates, SpecialMove):
        self.Width = GridWidth
        self.Height = GridHeight
        self.Actions = Actions
        self.nS = GridHeight * GridWidth
        self.nA = len(Actions)
        self.SpecialReward = SpecialReward
        self.EndStates = EndStates
        self.SpecialMove = SpecialMove
        self.P = self.__init_states(Prob, StepReward)

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
                    if (prob.value == 0.0):
                        continue
                    s_next = self.__generate_transation(
                        s, x, y, action + dir - 1)    # 处理每一个转移概率，方向逆时针减1
                    reward = StepReward              # 通用奖励定义 (-1)
                    if (s, s_next) in self.SpecialReward:    # 如果有特殊奖励定义
                        reward = self.SpecialReward[(s, s_next)]
                    list_probs.append((prob.value, s_next, reward))
                
                P[s][action] = list_probs
        return P

    # 左上角为 [0,0], 横向为 x, 纵向为 y
    def __generate_transation(self, s, x, y, action):
        action = action % 4         # 避免负数
        if (s,action) in self.SpecialMove:
            return self.SpecialMove[(s,action)]

        if (action == UP):          # 向上转移
            if (y != 0):            # 在上方边界处
                s = s - self.Width
        elif (action == DOWN):      # 向下转移
            if (y != self.Height-1):# 在下方边界处
                s = s + self.Width
        elif (action == LEFT):      # 向左转移
            if (x != 0):            # 在左侧边界处
                s = s - 1
        elif (action == RIGHT):     # 向右转移
            if (x != self.Width-1): # 在右侧边界处
                s = s + 1
        return s


    def step(self, s):
        pass

    def get_actions(self, s):
        actions = self.P[s]
        return actions.items()

def V_pi(env: GridWorld, gamma):
    V = np.zeros(env.nS)
    Q = copy.deepcopy(env.P)    # 拷贝状态转移结构以存储Q(s,a)值
    count = 0
    # 迭代
    while (count < 10):
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
                        q_pi += p * (r + gamma * V[s_next])
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
        print(np.reshape(np.round(V,2), (4,4)))
        count += 1
    # end while
    print(count)
    print(Q)
    return V

def V_pi_2array(env: GridWorld, gamma, iteration):
    V = np.zeros(env.nS)
    Q = copy.deepcopy(env.P)    # 拷贝状态转移结构以存储Q(s,a)值
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
    return V


def V_star(env: GridWorld, gamma):
    V_star = np.zeros(env.nS)
    Q_star = copy.deepcopy(env.P)    # 拷贝状态转移结构以存储Q(s,a)值
    count = 0
    # 迭代
    while (True):
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

action_names = ['LEFT', 'UP', 'RIGHT', 'DOWN']

def print_P(P):
    for s,v in P.items():
        print("state =",s)
        for action,v2 in v.items():
            print("\taction=", action_names[action])
            print("\t",v2)

chars = [0x2190, 0x2191, 0x2192, 0x2193]

def print_policy(policy, shape):
    best_actions = np.argmax(policy, axis=1)
    for i, action in enumerate(best_actions):
        print(chr(chars[action]), end="")
        print(" ", end="")
        if ((i+1) % shape[0] == 0):
            print("")
