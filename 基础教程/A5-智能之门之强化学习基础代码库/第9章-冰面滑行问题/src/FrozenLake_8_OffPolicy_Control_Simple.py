from enum import Enum
import common.Algo_DP_ValueIteration as algoDP
import numpy as np
import common.CommonHelper as helper
import gymnasium as gym
import common.Algo_MC_OnPolicy_Control as algoMCOnPolicy
import common.Algo_MC_OffPolicy_Control as algoMCOffPolicy
import copy


# 状态空间
class States(Enum):
     S0 = 0      # 开始
     S1 = 1 
     S2 = 2 
     S3 = 3
     T = 4       # 终止

# 动作空间
class Actions(Enum):
    a0 = 0     # 射击红色小气球
    a1 = 1    # 射击蓝色大气球

P = {
    # state: {action: [(p, s', r),...]}
    States.S0.value:{                                
        Actions.a0.value:[                           
            (0.7, States.S1.value, 0, False),
            (0.3, States.S2.value, 0, False),
        ],       
        Actions.a1.value:[   
            (0.1, States.S1.value, 0, False),
            (0.9, States.S2.value, 0, False),
        ]                   
    },
    States.S1.value:{ 
        Actions.a0.value:[
            (1.0, States.T.value, 3, True),
        ],
        Actions.a1.value:[                 
            (1.0, States.S3.value, 0, False),
        ]
    },
    States.S2.value:{
        Actions.a0.value:[                         
            (1.0, States.S3.value, 0, False),
        ],
        Actions.a1.value:[            
            (1.0, States.T.value, 2, True),
        ]
    },
    States.S3.value:{
        Actions.a0.value:[                         
            (1.0, States.T.value, 4, True),
        ],
        Actions.a1.value:[            
            (1.0, States.T.value, 1, True),
        ]
    },
    States.T.value:{                                 # 第一枪打红球误中蓝球
        Actions.a0.value:[                                 # 继续选择射击红球
            (1.0, States.T.value, 0, True),
        ],
        Actions.a1.value:[            
            (1.0, States.T.value, 0, True),
        ]
    }

}

class Env(object):
    def __init__(self, policy=None):
        self.observation_space = gym.spaces.Discrete(len(States))
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.S = States
        self.nS = len(States)
        self.A = Actions
        self.nA = len(Actions)
        self.P = P
        self.Policy = policy
        self.end_states = [States.T.value]

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.state = States.S0.value
        return States.S0.value, 0

    def get_actions(self, s):
        actions = self.P[s]
        return actions.items()

    def step(self, a):
        list_PSRE = self.P[self.state][a]
        list_P = [item[0] for item in list_PSRE]
        index = np.random.choice(len(list_PSRE), p=list_P)
        p, s_next, r, is_end = list_PSRE[index]
        self.state = s_next
        return s_next, r, is_end, None, None

    def is_end(self,s):
        if s in self.end_states:
            return True
        else:
            return False


if __name__=="__main__":
    behavior_policy = np.array([
        [0.5, 0.5],    # 在状态 0 时，选择红球的概率0.4，选择蓝球的概率0.6
        [0.5, 0.5],    # 在状态 1 时，同上
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    gamma = 1
    episodes = 1000

    helper.print_seperator_line(helper.SeperatorLines.long, "DP 价值迭代")
    env = Env()
    V_dp, Q_dp = algoDP.calculate_VQ_star(env, gamma)    # 迭代计算V,Q
    helper.print_V(V_dp, 3, (1,5), helper.SeperatorLines.middle, "V 值")
    helper.print_Q(Q_dp, 3, (1,5), helper.SeperatorLines.middle, "Q 值")
    V0 = copy.deepcopy(V_dp)
    Q0 = copy.deepcopy(Q_dp)
    V0[V0==0] = 1
    Q0[Q0==0] = 1
    policy = algoDP.get_policy_from_Q(env, Q_dp)
    helper.print_Policy(policy, 3, (1,5), helper.SeperatorLines.middle, "最优策略")

    # ======================
    
    helper.print_seperator_line(helper.SeperatorLines.long, "MC 异策略")
    ctrl = algoMCOffPolicy.MC_OffPolicy_Control_Policy(
        env, episodes, gamma, behavior_policy)
    Q2, target_policy = ctrl.run()
    helper.print_Policy(target_policy, 3, (1,5), helper.SeperatorLines.middle, "最优策略")
    helper.print_Q(Q2, 3, (1, 5), helper.SeperatorLines.middle, "Q 值")

