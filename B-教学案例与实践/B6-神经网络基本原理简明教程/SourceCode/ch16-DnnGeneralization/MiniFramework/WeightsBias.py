# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

from MiniFramework.Optimizer import *

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2,
    MSRA = 3

class WeightsBias(object):
    def __init__(self, n_input, n_output, init_method, optimizer_name, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.eta = eta
        self.initial_value_filename = str.format("w_{0}_{1}_{2}_init.npy", self.num_output, self.num_input, self.init_method.name)
        self.result_value_filename = str.format("{0}_{1}_{2}_result.npy", self.num_output, self.num_input, self.init_method.name)

    def InitializeWeights(self, create_new = False):
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        # end if
        self.__CreateOptimizers()

        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def __CreateNew(self):
        self.W, self.B = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.__SaveInitialValue()
        
    def __LoadExistingParameters(self):
        w_file = Path(self.initial_value_filename)
        if w_file.exists():
            self.__LoadInitialValue()
        else:
            self.__CreateNew()
        # end if

    def __CreateOptimizers(self):
        self.oW = OptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)
        self.oB = OptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)

    def pre_Update(self):
        if self.optimizer_name == OptimizerName.Nag:
            self.W = self.oW1.pre_update(self.W)
            self.B = self.oB1.pre_update(self.B)
        # end if

    def Update(self):
        self.W = self.oW.update(self.W, self.dW)
        self.B = self.oB.update(self.B, self.dB)

    def __SaveInitialValue(self):
        np.save(self.initial_value_filename, self.W)

    def __LoadInitialValue(self):
        self.W = np.load(self.initial_value_filename)
        self.B = np.zeros((self.num_output, 1))

    def SaveResultValue(self, name):
        np.save(name + "_w_" + self.result_value_filename, self.W)
        np.save(name + "_b_" + self.result_value_filename, self.B)

    def LoadResultValue(self, name):
        self.W = np.load(name + "_w_" + self.result_value_filename)
        self.B = np.load(name + "_b_" + self.result_value_filename)

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == InitialMethod.Zero:
            # zero
            W = np.zeros((num_output, num_input))
        elif method == InitialMethod.Normal:
            # normalize
            W = np.random.normal(size=(num_output, num_input))
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2/num_input), size=(num_output, num_input))
        elif method == InitialMethod.Xavier:
            # xavier
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                  np.sqrt(6/(num_output+num_input)),
                                  size=(num_output,num_input))
        # end if
        B = np.zeros((num_output, 1))
        return W, B

