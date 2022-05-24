import numpy as np
from enum import Enum

class SeperatorLines(Enum):
    empty = 0   # 打印空行
    short = 1   # 打印10个'-'
    middle = 2  # 打印30个'-'
    long = 3    # 打印40个'='

def print_seperator_line(style: SeperatorLines):
    if style == SeperatorLines.empty:
        print("")
    elif style == SeperatorLines.short:
        print("-"*10)
    elif style == SeperatorLines.middle:
        print("-"*30)
    elif style == SeperatorLines.long:
        print("="*40)

def print_V(dataModel, V):
    vv = np.around(V,2)
    print("状态价值函数计算结果(数组) :", vv)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))
