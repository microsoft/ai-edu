# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

from GDOptimizer import *

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2

class WeightsBias(object):
    def __init__(self, n_input, n_output, init_method, optimizer_name, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.eta = eta

    def __GenerateWeightsArrayFileName(self):
        self.w_filename = str.format("w1_{0}_{1}_{2}.npy", self.num_output, self.num_input, self.init_method.name)

    def InitializeWeights(self, create_new = False):
        self.__GenerateWeightsArrayFileName()
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
        np.save(self.w_filename, self.W)
        
    def __LoadExistingParameters(self):
        w_file = Path(self.w_filename)
        if w_file.exists() and w_file.exists():
            self.W = np.load(w_file)
            self.B = np.zeros((self.num_output, 1))
        else:
            self.__CreateNew()
        # end if

    def __CreateOptimizers(self):
        self.oW = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)
        self.oB = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)

    def pre_Update(self):
        if self.optimizer_name == OptimizerName.Nag:
            self.W = self.oW1.pre_update(self.W)
            self.B = self.oB1.pre_update(self.B)
        # end if

    def Update(self):
        self.W = self.oW.update(self.W, self.dW)
        self.B = self.oB.update(self.B, self.dB)

    def GetWeightsBiasAsDict(self):
        dict = {"W":self.W, "B":self.B}
        return dict

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == InitialMethod.Zero:
            # zero
            W = np.zeros((num_output, num_input))
        elif method == InitialMethod.Normal:
            # normalize
            W = np.random.normal(size=(num_output, num_input))
        elif method == InitialMethod.Xavier:
            # xavier
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                  np.sqrt(6/(num_output+num_input)),
                                  size=(num_output,num_input))
        # end if
        B = np.zeros((num_output, 1))
        return W, B

