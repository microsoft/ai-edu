# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

class CBookmark(object):
    def __init__(self, loss, weights, epoch, iteration):
        self.loss = loss
        self.weights = weights
        self.epoch = epoch
        self.iteration = iteration
    # end def
# end class

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class CLossHistory(object):
    def __init__(self):
        # loss history
        self.loss_history = []
        self.min_loss_index = -1
        self.min_bookmark = CBookmark(100000, None, -1, -1)

    def AddLossHistory(self, loss, weights, epoch, iteration):
        self.loss_history.append(loss)
        if loss < self.min_bookmark.loss:
            self.min_bookmark = CBookmark(loss, weights, epoch, iteration)
            self.minimal_loss_index = len(self.loss_history) - 1
        # end if

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, title):
        plt.plot(self.loss_history)
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()

        # 从历史记录中获得最小损失值得训练权重值
    def GetMinimalLossData(self):
        return self.min_bookmark

# end class
