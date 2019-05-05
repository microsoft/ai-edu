# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

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

# help function to record loss history
class CLossHistory(object):
    def __init__(self):
        # loss history
        self.loss_history = []
        self.min_loss_index = -1
        # initialize with a big number, will be overwriten by real number
        self.min_trace = CTrace(100000, -1, -1, None, None)

    def AddLossHistory(self, loss, epoch, iteration, wb1, wb2):
        self.loss_history.append(loss)
        if loss < self.min_trace.loss:
            self.min_trace = CTrace(loss, epoch, iteration, wb1, wb2)
            self.minimal_loss_index = len(self.loss_history) - 1
            return True
        # end if
        return False

    # 
    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.loss_history)
        title = self.min_trace.toString() + "," + params.toString()
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
        return title

        # get minimal trace
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
