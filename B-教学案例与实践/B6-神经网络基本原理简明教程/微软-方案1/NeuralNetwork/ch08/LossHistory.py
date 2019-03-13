# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.



class CBookmark(object):
    def _init__(self, loss, weights, epoch, iteration):
        self.loss = loss
        self.weights = weights
        self.epoch = epoch
        self.iteration = iteration

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class CLossHistory(object):
    def __init__(self):
        self.loss_history = []
        self.mininal_loss = 100
        self.minimal_loss_position = -1

    def __init__(self, loss, weights, epoch, iteration):
        bookmark = CBookmark(loss, weights, epoch, iteration)


        self.dict_loss = {}  # loss history

        self.loss = loss
        self.dict_weights = dict_weights
        self.epoch = epoch
        self.iteration = iteration

    def AddLossHistory(self, loss, weights, epoch, iteration):


        # 图形显示损失函数值历史记录
    def ShowLossHistory(self, dict_loss, method):
        loss = []
        for key in dict_loss:
            loss.append(key)

        #plt.plot(loss)
        plt.plot(loss)
        plt.title(method)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()

        # 从历史记录中获得最小损失值得训练权重值
    def GetMinimalLossData(self, dict_loss):
        key = sorted(dict_loss.keys())[0]
        dict_weights = dict_loss[key].dict_weights
        return dict_weights, dict_loss[key]
