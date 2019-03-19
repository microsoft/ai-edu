# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

class CParameters(object):
    def __init__(self, n_example, n_input=1, n_output=1, n_hidden=4, eta=0.1, max_epoch=10000, batch_size=5, lossFunType="MSE", eps=0.001, initMethod=0):
        self.num_example = n_example
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
        self.loss_func_type = lossFunType
        self.eps = eps
        self.init_method = initMethod

def InitialParameters(num_input, num_output, method, loadExist=False):

    if loadExist == True:
        dict_exist = np.load("weights_bias.npy")
        # assume there have W1,W2,B1,B2 in this dictionary
        W1 = dict_exist["W1"]
        W2 = dict_exist["W2"]
        B1 = dict_exist["B1"]
        B2 = dict_exist["B2"]

    # end if

    if method == 0:
        # zero
        W = np.zeros((num_output, num_input))
    elif method == 1:
        # normalize
        W = np.random.normal(size=(num_output, num_input))
    elif method == 2:
        # xavier
        W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                              np.sqrt(6/(num_output+num_input)),
                              size=(num_output,num_input))
    # end if
    B = np.zeros((num_output, 1))
    return W, B


