# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
from enum import Enum
from LossFunction import LossFunctionName

class InitialMethod(Enum):
    zero = 0,
    norm = 1,
    xavier = 2

# this class is for two-layer NN only
class CParameters(object):
    def __init__(self, n_example, n_input=1, n_output=1, n_hidden=4, eta=0.1, max_epoch=10000, batch_size=5, lossFuncName=LossFunctionName.MSE, eps=0.001, initMethod=InitialMethod.zero):
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
        self.loss_func_name = lossFuncName
        self.eps = eps
        self.init_method = initMethod

    def GenerateWeightsArrayFileName(self):
        self.w1_name = str.format("w1_{0}_{1}_{2}.npy", self.num_hidden, self.num_input, self.init_method.name)
        self.w2_name = str.format("w2_{0}_{1}_{2}.npy", self.num_output, self.num_hidden, self.init_method.name)

    # load exist initial weights which has same parameters
    # if not found, create new, then save
    def LoadSameInitialParameters(self):
        self.GenerateWeightsArrayFileName()
        w1_file = Path(self.w1_name)
        w2_file = Path(self.w2_name)
        if w1_file.exists() and w2_file.exists():
            W1 = np.load(w1_file)
            W2 = np.load(w2_file)
            B1 = np.zeros((self.num_hidden, 1))
            B2 = np.zeros((self.num_output, 1))
        else:
            W1, B1 = CParameters.InitialParameters(self.num_input, self.num_hidden, self.init_method)
            W2, B2 = CParameters.InitialParameters(self.num_hidden, self.num_output, self.init_method)
            np.save(w1_file, W1)
            np.save(w2_file, W2)
        # end if
        return W1, B1, W2, B2

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == InitialMethod.zero:
            # zero
            W = np.zeros((num_output, num_input))
        elif method == InitialMethod.norm:
            # normalize
            W = np.random.normal(size=(num_output, num_input))
        elif method == InitialMethod.xavier:
            # xavier
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                  np.sqrt(6/(num_output+num_input)),
                                  size=(num_output,num_input))
        # end if
        B = np.zeros((num_output, 1))
        return W, B


