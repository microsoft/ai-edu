# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from enum import Enum

from MiniFramework.LossFunction import *
from MiniFramework.WeightsBias import *
from MiniFramework.Optimizer import *

class RegularMethod(Enum):
    L0 = 0,
    L1 = 1,
    L2 = 2,
    EarlyStop = 4


# this class is for two-layer NN only
class CParameters(object):
    def __init__(self, eta=0.1, max_epoch=10000, batch_size=5, eps = 0.1,
                 lossFuncName=LossFunctionName.MSE, 
                 initMethod=InitialMethod.Zero, 
                 optimizerName=OptimizerName.SGD,
                 regularName=RegularMethod.L0, lambd=0.0):

        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        if batch_size == -1:
            self.batch_size = self.num_example
        else:
            self.batch_size = batch_size
        # end if
        self.eps = eps
        self.loss_func = lossFuncName
        self.init_method = initMethod
        self.optimizer = optimizerName
        self.regular = regularName
        self.lambd = lambd


    def toString(self):
        title = str.format("bz:{0},eta:{1},init:{2},op:{3}", self.batch_size, self.eta, self.init_method.name, self.optimizer.name)
        if self.regular != RegularMethod.L0:
            title += str.format(",rgl:{0},lambd:{1}", self.regular.name, self.lambd)
        return title

