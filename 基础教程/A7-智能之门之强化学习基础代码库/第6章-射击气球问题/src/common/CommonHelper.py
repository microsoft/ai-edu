
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
