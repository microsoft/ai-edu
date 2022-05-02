import numpy as np
from enum import Enum
import copy

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

class GridWorld(object):
    # 生成环境
    def __init__(self, GridHeight, GridWidth, Actions, EndStatesWithReward, Prob, StepReward):
        self.nS = GridHeight * GridWidth
        self.nA = len(Actions)
        self.E = EndStatesWithReward
        self.P = self.__init_states(GridHeight, GridWidth, Actions, Prob, StepReward)

    def __init_states(self, H, W, A, Prob, StepReward):
        P = {}
        s_id = 0
        Pos2Sid = {}
        Sid2Pos = {}
        for y in range(H):
            for x in range(W):
                Pos2Sid[x,y] = s_id
                Sid2Pos[s_id] = [x,y]
                s_id += 1

        for s, (x,y) in Sid2Pos.items():
            P[s] = {}
            if ((x,y) in self.E):
                continue
            for action in A:
                list_probs = []
                for dir, prob in enumerate(Prob):
                    if (prob.value == 0.0):
                        continue
                    s_next, x_next, y_next = self.__generate_transation(
                        s, x, y, W, H, action.value + dir - 1)
                    reward = StepReward              # 通用奖励
                    if (x_next,y_next) in self.E:    # 如果有特殊定义
                        reward = self.E[(x_next,y_next)]
                    list_probs.append((prob.value, s_next, reward))
                
                P[s][action] = list_probs
        return P

    # 左上角为 [0,0],
    def __generate_transation(self, s, x, y, MAX_X, MAX_Y, action):
        action = action % 4     # 避免负数
        if (action == UP):      # 向上移动
            if (y == 0):        # 在上方边界处
                return s, x, y  # 原地不动
            else:
                s = s - MAX_X
                y = y - 1
        elif (action == DOWN):  # 向下移动
            if (y == MAX_Y-1):  # 在下方边界处
                return s, x, y  # 原地不动
            else:
                s = s + MAX_X
                y = y + 1
        elif (action == LEFT):  # 向左移动
            if (x == 0):        # 在左侧边界处
                return s, x, y  # 原地不动
            else:
                s = s - 1
                x = x - 1
        elif (action == RIGHT): # 向右移动
            if (x == MAX_X-1):  # 在右侧边界处
                return s, x, y  # 原地不动
            else:
                s = s + 1
                x = x + 1
        return s, x, y


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

