# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 4.0
what's new?
- remove hidden definition
"""

from MiniFramework.EnumDef_3_0 import *

# this class is for two-layer NN only
class HyperParameters_4_0(object):
    def __init__(self, eta=0.1, max_epoch=10000, batch_size=5,
                 net_type=NetType.Fitting, 
                 init_method=InitialMethod.Xavier,
                 stopper = None):
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
        self.stopper = stopper

    def toString(self):
        title = str.format("bz:{0},eta:{1},init:{2}", self.batch_size, self.eta, self.init_method.name)
        return title
