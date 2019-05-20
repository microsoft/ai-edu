# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class CTrace(object):
    def __init__(self, epoch, iteration, loss, accuracy):
        self.loss = loss
        self.epoch = epoch
        self.iteration = iteration
        self.accuracy = accuracy
    # end def

    def toString(self):
        info = str.format("epc={0},ite={1},los={2:.4f},acy={2:.4f}", self.epoch, self.iteration, self.loss, self.accuracy)
        return info

# end class

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class CLossHistory(object):
    def __init__(self, need_earlyStop = False, patience = 5):
        # loss history
        self.loss_history_train = []
        self.accuracy_history_train = []
        self.iteration_history_train = []
        self.epoch_history_train = []

        self.loss_history_val = []
        self.accuracy_history_val = []
       
        # for early stop
        self.min_loss_index = -1
        self.early_stop = need_earlyStop
        self.patience = patience
        self.patience_counter = 0
        self.last_vld_loss = float("inf")

    def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
        self.iteration_history_train.append(total_iteration)
        self.epoch_history_train.append(epoch)
        self.loss_history_train.append(loss_train)
        self.accuracy_history_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_history_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_history_val.append(accuracy_vld)

        if self.early_stop:
            if loss_vld < self.last_vld_loss:
                self.patience_counter = 0
                self.last_vld_loss = loss_vld
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    return True     # need to stop
            # end if
        return False

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        fig = plt.figure(figsize=(12,5))

        axes = plt.subplot(1,2,1)
        p2, = axes.plot(self.iteration_history_train, self.loss_history_train)
        p1, = axes.plot(self.iteration_history_train, self.loss_history_val)
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Loss")
        axes.set_ylabel("loss")
        axes.set_xlabel("iteration")
        if xmin != None or xmax != None or ymin != None or ymax != None:
            axes.axis([xmin, xmax, ymin, ymax])

        
        axes = plt.subplot(1,2,2)
        p2, = axes.plot(self.iteration_history_train, self.accuracy_history_train)
        p1, = axes.plot(self.iteration_history_train, self.accuracy_history_val)
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Accuracy")
        axes.set_ylabel("accuracy")
        axes.set_xlabel("iteration")
        
        title = params.toString()
        plt.suptitle(title)
        plt.show()
        return title
        """
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
        return title
        """
        # 从历史记录中获得最小损失值得训练权重值
    def GetMinimalLossData(self):
        return self.min_trace

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
