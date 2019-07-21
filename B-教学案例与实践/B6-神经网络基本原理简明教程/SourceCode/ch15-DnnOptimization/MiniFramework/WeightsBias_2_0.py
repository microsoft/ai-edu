# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

from MiniFramework.EnumDef_4_0 import *
from MiniFramework.Optimizer_1_0 import *

class WeightsBias_2_0(object):
    def __init__(self, n_input, n_output, init_method, optimizer_name, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.eta = eta
        self.optimizer_name = optimizer_name
        self.initial_value_filename = str.format("w_{0}_{1}_{2}_init", self.num_input, self.num_output, self.init_method.name)

    def InitializeWeights(self, folder, create_new):
        self.folder = folder
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        # end if
        self.__CreateOptimizers()

        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def __CreateNew(self):
        self.W, self.B = WeightsBias_2_0.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.__SaveInitialValue()
        
    def __LoadExistingParameters(self):
        file_name = str.format("{0}\\{1}.npz", self.folder, self.initial_value_filename)
        w_file = Path(file_name)
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
            self.W = self.oW.pre_update(self.W)
            self.B = self.oB.pre_update(self.B)
        # end if

    def Update(self):
        self.W = self.oW.update(self.W, self.dW)
        self.B = self.oB.update(self.B, self.dB)

    def __SaveInitialValue(self):
        file_name = str.format("{0}\\{1}.npz", self.folder, self.initial_value_filename)
        np.savez(file_name, weights=self.W, bias=self.B)

    def __LoadInitialValue(self):
        file_name = str.format("{0}\\{1}.npz", self.folder, self.initial_value_filename)
        data = np.load(file_name)
        self.W = data["weights"]
        self.B = data["bias"]

    def SaveResultValue(self, folder, name):
        file_name = str.format("{0}\\{1}.npz", folder, name)
        np.savez(file_name, weights=self.W, bias=self.B)

    def LoadResultValue(self, folder, name):
        file_name = str.format("{0}\\{1}.npz", folder, name)
        data = np.load(file_name)
        self.W = data["weights"]
        self.B = data["bias"]

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == InitialMethod.Zero:
            # zero
            W = np.zeros((num_input, num_output))
        elif method == InitialMethod.Normal:
            # normalize
            W = np.random.normal(size=(num_input, num_output))
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2/num_output), size=(num_input, num_output))
        elif method == InitialMethod.Xavier:
            # xavier
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                  np.sqrt(6/(num_output+num_input)),
                                  size=(num_input, num_output))
        # end if
        B = np.zeros((1, num_output))
        return W, B

