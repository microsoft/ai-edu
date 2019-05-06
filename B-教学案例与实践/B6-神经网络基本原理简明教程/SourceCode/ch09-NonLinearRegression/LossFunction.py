# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import pickle

class CTrace(object):
    def __init__(self, loss, epoch, iteration, wb1, wb2):
        self.loss = loss
        self.epoch = epoch
        self.iteration = iteration
        self.wb1 = wb1
        self.wb2 = wb2
    # end def

    def toString(self):
        info = str.format("epc={0},ite={1},los={2:.4f}", self.epoch, self.iteration, self.loss)
        return info

# end class

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class CLossHistory(object):
    def __init__(self, params):
        # loss history
        self.loss_history = []
        self.min_loss_index = -1
        # 初始化一个极大值,在后面的肯定会被更小的loss值覆盖
        self.min_trace = CTrace(100000, -1, -1, None, None)
        self.params = params

    def AddLossHistory(self, loss, epoch, iteration, wb1, wb2):
        self.loss_history.append(loss)
        if loss < self.min_trace.loss:
            self.min_trace = CTrace(loss, epoch, iteration, wb1, wb2)
            self.minimal_loss_index = len(self.loss_history) - 1
            return True
        # end if
        return False

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.loss_history)
        title = self.min_trace.toString() + "," + self.params.toString()
        plt.subtitle(title)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
        return title

    def ShowLossHistory(self, axes, xmin=None, xmax=None, ymin=None, ymax=None):
        axes.plot(self.loss_history)
        title = self.min_trace.toString() + "," + self.params.toString()
        axes.set_title(title)
        axes.set_xlabel("epoch")
        axes.set_ylabel("loss")
        if xmin != None and ymin != None:
            axes.axis([xmin, xmax, ymin, ymax])
        #plt.show()
        return title

        # 从历史记录中获得最小损失值得训练权重值
    def GetMinimalLossData(self):
        return self.min_trace

    def Dump(self, name):
        f = open(name, 'wb')
        pickle.dump(self, f)

    def Load(name):
        f = open(name, 'rb')
        lh = pickle.load(f)
        return lh

    def SaveToArray(self, name):
        arr = np.array(self.loss_history)
        np.save(name, arr)

# end class

class LossFunctionName(Enum):
    MSE = 1,
    CrossEntropy2 = 2,
    CrossEntropy3 = 3,

class CLossFunction(object):
    def __init__(self, func_name):
        self.func_name = func_name
    # end def

    # fcFunc: feed forward calculation
    def CheckLoss(self, Y, A):
        m = Y.shape[1]
        if self.func_name == LossFunctionName.MSE:
            loss = self.MSE(A, Y, m)
        elif self.func_name == LossFunctionName.CrossEntropy2:
            loss = self.CE2(A, Y, m)
        elif self.func_name == LossFunctionName.CrossEntropy3:
            loss = self.CE3(A, Y, m)
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
        p1 = np.log(A+1e-7)
        p2 =  np.multiply(Y, p1)
        LOSS = np.sum(-p2) 
        loss = LOSS / count
        return loss
    # end def
