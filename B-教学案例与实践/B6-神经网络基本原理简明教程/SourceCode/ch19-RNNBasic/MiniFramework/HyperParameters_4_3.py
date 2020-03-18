# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *

# this class is for RNN only
class HyperParameters_4_3(object):
    def __init__(self, eta, max_epoch, batch_size,
                 num_step, num_input, num_hidden, num_output,
                 output_type=OutputType.EachStep,
                 net_type=NetType.Fitting, 
                 init_method=InitialMethod.Xavier,
                 optimizer_name=OptimizerName.SGD):
        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        if batch_size == -1:
            self.batch_size = self.num_example
        else:
            self.batch_size = batch_size
        # end if
        self.net_type = net_type
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.num_step = num_step
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.output_type = output_type

    def toString(self):
        #title = str.format("bz:{0},eta:{1},init:{2},op:{3}", self.batch_size, self.eta, self.init_method.name, self.optimizer_name.name)
        title = str.format("epoch:{0},batch:{1},hidden:{2},eta:{3}", self.max_epoch, self.batch_size, self.num_hidden, self.eta)
        return title

class HyperParameters_4_4(object):
    def __init__(self, eta, max_epoch, batch_size,
                 num_step, num_input, num_hidden1, num_hidden2, num_output,
                 net_type=NetType.Fitting, 
                 init_method=InitialMethod.Xavier,
                 optimizer_name=OptimizerName.SGD):
        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        if batch_size == -1:
            self.batch_size = self.num_example
        else:
            self.batch_size = batch_size
        # end if
        self.net_type = net_type
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.num_step = num_step
        self.num_input = num_input
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_output = num_output

    def toString(self):
        title = str.format("bz:{0},eta:{1},init:{2},op:{3}", self.batch_size, self.eta, self.init_method.name, self.optimizer_name.name)
        return title
