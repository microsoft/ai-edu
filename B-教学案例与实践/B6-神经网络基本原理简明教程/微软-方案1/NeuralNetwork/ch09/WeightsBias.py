# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

from GDOptimizer import *

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2

# this class is designed for 2 layers
class WeightsBias(object):
    def __init__(self, params):
        self.num_input = params.num_input
        self.num_hidden = params.num_hidden
        self.num_output = params.num_output
        self.init_method = params.init_method
        self.optimizer_name = params.optimizer_name
        self.eta = params.eta

    def GenerateWeightsArrayFileName(self):
        self.w1_filename = str.format("w1_{0}_{1}_{2}.npy", self.num_hidden, self.num_input, self.init_method.name)
        self.w2_filename = str.format("w2_{0}_{1}_{2}.npy", self.num_output, self.num_hidden, self.init_method.name)

    def InitializeWeights(self, create_new = False):
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        # end if
        self.__CreateOptimizers()

    def __CreateNew(self):
        self.W1, self.B1 = WeightsBias.InitialParameters(self.num_input, self.num_hidden, self.init_method)
        self.W2, self.B2 = WeightsBias.InitialParameters(self.num_hidden, self.num_output, self.init_method)
        self.dW1 = np.zeros(self.W1.shape)
        self.dB1 = np.zeros(self.B1.shape)
        self.dW2 = np.zeros(self.W2.shape)
        self.dB2 = np.zeros(self.B2.shape)
        self.GenerateWeightsArrayFileName()
        np.save(self.w1_filename, self.W1)
        np.save(self.w2_filename, self.W2)
        

    def __LoadExistingParameters(self):
        self.GenerateWeightsArrayFileName()
        w1_file = Path(self.w1_filename)
        w2_file = Path(self.w2_filename)
        if w1_file.exists() and w2_file.exists():
            self.W1 = np.load(w1_file)
            self.W2 = np.load(w2_file)
            self.B1 = np.zeros((self.num_hidden, 1))
            self.B2 = np.zeros((self.num_output, 1))
        else:
            self.__CreateNew()
        # end if

    def __CreateOptimizers(self):
        self.oW1 = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)
        self.oB1 = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)
        self.oW2 = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)
        self.oB2 = GDOptimizerFactory.CreateOptimizer(self.eta, self.optimizer_name)

    def pre_Update(self):
        if self.optimizer_name == OptimizerName.Nag:
            self.W1 = self.oW1.pre_update(self.W1)
            self.B1 = self.oB1.pre_update(self.B1)
            self.W2 = self.oW2.pre_update(self.W2)
            self.B2 = self.oB2.pre_update(self.B2)
        # end if

    def Update(self):
        self.W1 = self.oW1.update(self.W1, self.dW1)
        self.B1 = self.oB1.update(self.B1, self.dB1)
        self.W2 = self.oW2.update(self.W2, self.dW2)
        self.B2 = self.oB2.update(self.B2, self.dB2)

    def GetWeightsBiasAsDict(self):
        dict = {"W1":self.W1, "B1":self.B1, "W2":self.W2, "B2":self.B2}
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

