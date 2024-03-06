import numpy as np
from enum import Enum


class SeperatorLines(Enum):
    empty = 0   # 打印空行
    short = 1   # 打印10个'-'
    middle = 2  # 打印20个'-'
    long = 3    # 打印40个'='

def print_seperator_line(style: SeperatorLines):
    if style == SeperatorLines.empty:
        print("")
    elif style == SeperatorLines.short:
        print("-"*10)
    elif style == SeperatorLines.middle:
        print("-"*20)
    elif style == SeperatorLines.long:
        print("="*40)

def extract_policy_from_Q(Q):
    policy = []
    i = 0
    for s in range(Q.shape[0]):
        max_v = np.max(Q[s])
        best_actions = np.argwhere(Q[s] == max_v).flatten()
        policy.append(best_actions)
        print(best_actions)
        # for a in range(Q[s].shape[0]):
        #     if Q[s,a] == max_v:
        #         policy[s, a] = 1
    return policy
