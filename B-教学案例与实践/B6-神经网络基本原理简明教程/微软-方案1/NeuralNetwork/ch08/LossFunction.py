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

    def print_info(self):
        print("epoch=%d, iteration=%d, loss=%f" %(self.epoch, self.iteration, self.loss))

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
    def ShowLossHistory(self, params):
        plt.plot(self.loss_history)
        title = str.format("los:{0:.5f} ep:{1} ite:{2} bz:{3} eta:{4} ne:{5}", 
                           self.min_bookmark.loss, self.min_bookmark.epoch, self.min_bookmark.iteration, 
                           params.batch_size, params.eta, params.num_hidden)
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()

        # 从历史记录中获得最小损失值得训练权重值
    def GetMinimalLossData(self):
        return self.min_bookmark

# end class

class CLossFunction(object):
    def __init__(self, funcType):
        self.func_type = funcType
    # end def

    # fcFunc: feed forward calculation
    def CheckLoss(self, X, Y, dict_weights, ffcFunc):
        m = X.shape[1]
        dict_cache = ffcFunc(X, dict_weights)
        output = dict_cache["Output"]
        if self.func_type == "MSE":
            loss = self.MSE(output, Y, m)
        elif self.func_type == "CE2":
            loss = self.CE2(output, Y, m)
        elif self.func_type == "CE3":
            loss = self.CE3(output, Y, m)
        #end if
        return loss
    # end def

    def MSE(self, A, Y, count):
        p1 = A - Y
        LOSS = np.multiply(p1, p1)
        loss = LOSS.sum()/count/2
        return loss
    # end def

    # for binary classifier
    def CE2(self, A, Y, count):
        p1 = 1 - Y
        p2 = np.log(1 - A)
        p3 = np.log(A)

        p4 = np.multiply(p1 ,p2)
        p5 = np.multiply(Y, p3)

        LOSS = np.sum(-(p4 + p5))  #binary classification
        loss = LOSS / count
        return loss
    # end def

    # for multiple classifier
    def CE3(self, A, Y, count):
        p1 = np.log(A)
        p2 =  np.multiply(Y, p1)
        LOSS = np.sum(-p2) 
        loss = LOSS / count
        return loss
    # end def
