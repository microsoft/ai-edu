# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
from enum import Enum

from GDOptimizer import *

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2,
    MSRA = 3

class WeightsBias(object):
    def __init__(self, n_input, n_output, eta, init_method = InitialMethod.Xavier, optimizer = OptimizerName.SGD):
        self.num_input = n_input
        self.num_output = n_output
        self.eta = eta
        self.init_method = init_method
        self.optimizer_name = optimizer

    def __GenerateWeightsArrayFileName(self):
        self.w_filename = str.format("w1_{0}_{1}_{2}.npy", self.num_output, self.num_input, self.init_method.name)

    def InitializeWeights(self, create_new = False):
        self.__GenerateWeightsArrayFileName()
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        # end if
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)
        self.__CreateOptimizer()

    def __CreateOptimizer(self):
        self.opt_W = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)
        self.opt_B = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)

    def __CreateNew(self):
        self.W, self.B = WeightsBias.InitialParameters(self.num_input, self.num_output, self.init_method)
        np.save(self.w_filename, self.W)
        
    def __LoadExistingParameters(self):
        w_file = Path(self.w_filename)
        if w_file.exists():
            self.W = np.load(w_file)
            self.B = np.zeros((self.num_output, 1))
        else:
            self.__CreateNew()
        # end if

    def pre_Update(self):
        self.W = self.opt_W.pre_update(self.W)
        self.B = self.opt_B.pre_update(self.B)

    def UpdateWithLR(self, lr):
        if lr != None:
            self.W = self.W - lr * self.dW
            self.B = self.B - lr * self.dB
        else:
            self.W = self.W - self.eta * self.dW
            self.B = self.B - self.eta * self.dB

    def Update(self):
        self.W = self.opt_W.update(self.W, self.dW)
        self.B = self.opt_B.update(self.B, self.dB)

    def GetWeightsBiasAsDict(self):
        dict = {"W":self.W, "B":self.B}
        return dict

    def Save(self, name):
        np.save(name+"W.npy", self.W)
        np.save(name+"B.npy", self.B)

    def Load(self, name):
        self.W = np.load(name+"W.npy")
        self.B = np.load(name+"B.npy")

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

