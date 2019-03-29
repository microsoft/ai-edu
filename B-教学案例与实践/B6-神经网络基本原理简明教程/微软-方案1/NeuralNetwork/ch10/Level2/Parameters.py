# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from LossFunction import *
from WeightsBias import *
from GDOptimizer import *

# this class is for two-layer NN only
class CParameters(object):
    def __init__(self, n_input=1, n_output=1, n_hidden=4, 
                 eta=0.1, max_epoch=10000, batch_size=5, eps=0.001,
                 lossFuncName=LossFunctionName.MSE, 
                 initMethod=InitialMethod.Zero, 
                 optimizerName=OptimizerName.SGD):

        self.num_input = n_input
        self.num_output = n_output
        self.num_hidden = n_hidden
        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        if batch_size == -1:
            self.batch_size = self.num_example
        else:
            self.batch_size = batch_size
        # end if
        self.loss_func_name = lossFuncName
        self.eps = eps
        self.init_method = initMethod
        self.optimizer_name = optimizerName

    def toString(self):
        title = str.format("bz:{0},eta:{1},ne:{2},op:{3}", self.batch_size, self.eta, self.num_hidden, self.optimizer_name.name)
        return title

