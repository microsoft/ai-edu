# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.LossFunction import *
from MiniFramework.WeightsBias import *
from MiniFramework.Optimizer import *

# this class is for two-layer NN only
class CParameters(object):
    def __init__(self, eta=0.1, max_epoch=10000, batch_size=5, eps = 0.1,
                 lossFuncName=LossFunctionName.MSE, 
                 initMethod=InitialMethod.Zero, 
                 optimizerName=OptimizerName.SGD):

        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        if batch_size == -1:
            self.batch_size = self.num_example
        else:
            self.batch_size = batch_size
        # end if
        self.loss_func_name = lossFuncName
        self.init_method = initMethod
        self.optimizer_name = optimizerName
        self.eps = eps


    def toString(self):
        title = str.format("bz:{0},eta:{1},init:{2},op:{3}", self.batch_size, self.eta, self.init_method.name, self.optimizer_name.name)
        return title

