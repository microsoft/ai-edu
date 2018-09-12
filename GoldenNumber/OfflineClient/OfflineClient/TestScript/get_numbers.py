# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# 为了“开箱即用”，本脚本没有依赖除了Python库以外的组件。
# 添加自己的代码时，可以自由地引用如numpy这样的组件以方便编程。

import sys
import itertools

def LineToNums(line, type=float):
    """将输入的行按间隔符分割，并转换为某种数字类型的可迭代对象（默认为float类型）"""
    return (type(cell) for cell in line.split('\t'))

metaLine = sys.stdin.readline()
lineNum, columnNum = LineToNums(metaLine, int)

history = []
for line in map(lambda _: sys.stdin.readline(), range(lineNum)):
    gnum, *nums = LineToNums(line)
    history.append((gnum, nums))

def Mean(iter, len):
    """用于计算均值的帮主函数"""
    return sum(iter) / len

if len(history) == 0:
    print("42\t4.2")
else:
    # 取最近的记录，最多五项。计算这些记录中黄金点的均值作为本脚本的输出。
    candidate1 = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5))
    candidate2 = candidate1 * 0.618 # 第二输出。

    print("%f\t%f" % (candidate1, candidate2))