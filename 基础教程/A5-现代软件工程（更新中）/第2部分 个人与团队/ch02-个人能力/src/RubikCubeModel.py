from enum import Enum

# 颜色
class Colors(Enum):
    White = 0
    Red = 1
    Yellow = 2
    Blue = 3
    Green = 4
    Orange = 5

# 位置, 以魔方的中心点为坐标(0,0,0), 每个块在[-2,+2]之间
class Position():
    x:int
    y:int
    z:int

# 块的基类
class Block():
    postion: Position

# 中心块, 一种颜色
class CenterBlock(Block):
    color_0: Colors

# 棱块, 两种颜色
class EdgeBlock(Block):
    color_0: Colors
    color_1: Colors

# 角块, 三种颜色
class CornerBlock(Block):
    color_0: Colors
    color_1: Colors
    color_2: Colors

# 一个面
class Side():
    # 属性
    center_block: CenterBlock      # 1个中心块
    corner_blocks: CornerBlock[4]  # 4个角块
    edge_blocks: EdgeBlock[4]      # 4个棱块
    # 动作
    def Rotate_90():  pass
    def Rotate1_80(): pass
    def Rotate_270(): pass

# 一个魔方
class RibukCube():
    side_Front: Side
    side_Back:  Side
    side_Up:    Side
    side_Down:  Side
    side_Left:  Side
    side_Right: Side
