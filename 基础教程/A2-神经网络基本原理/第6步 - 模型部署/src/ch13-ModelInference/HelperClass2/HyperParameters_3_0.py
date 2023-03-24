# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 3.0
what's new:
- add n_hidden1 and n_hidden2
"""

from HelperClass2.EnumDef_2_0 import *

# this class is for two-layer NN only
class HyperParameters_3_0(object):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, 
                 eta=0.1, max_epoch=10000, batch_size=5, eps = 0.1,
                 net_type = NetType.Fitting,
                 init_method = InitialMethod.Xavier):

        self.num_input = n_input
        self.num_hidden1 = n_hidden1
        self.num_hidden2 = n_hidden2
        self.num_output = n_output

        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        self.batch_size = batch_size  

        self.net_type = net_type
        self.init_method = init_method
        self.eps = eps

    def toString(self):
        title = str.format("bz:{0},eta:{1},ne:{2}x{3}", self.batch_size, self.eta, self.num_hidden1, self.num_hidden2)
        return title

