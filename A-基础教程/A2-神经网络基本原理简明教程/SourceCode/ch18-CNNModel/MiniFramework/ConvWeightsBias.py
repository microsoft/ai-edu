# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import math
from MiniFramework.EnumDef_6_0 import *
from MiniFramework.WeightsBias_2_1 import *
from MiniFramework.Optimizer_1_0 import *

class ConvWeightsBias(WeightsBias_2_1):
    def __init__(self, output_c, input_c, filter_h, filter_w, init_method, optimizer_name, eta):
        self.FilterCount = output_c
        self.KernalCount = input_c
        self.KernalHeight = filter_h
        self.KernalWidth = filter_w
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.learning_rate = eta

    def Initialize(self, folder, name, create_new):
        self.WBShape = (self.FilterCount, self.KernalCount, self.KernalHeight, self.KernalWidth)
        self.init_file_name = str.format(
            "{0}/{1}_{2}_{3}_{4}_{5}_init.npz", 
            folder, name, self.FilterCount, self.KernalCount, self.KernalHeight, self.KernalWidth)
        self.result_file_name = str.format("{0}/{1}_result.npz", folder, name)

        if create_new:
            self.CreateNew()
        else:
            self.LoadExistingParameters()

        # end if
        self.CreateOptimizers()

        self.dW = np.zeros(self.W.shape).astype('float32')
        self.dB = np.zeros(self.B.shape).astype('float32')

    def CreateNew(self):
        self.W = ConvWeightsBias.InitialConvParameters(self.WBShape, self.init_method)
        self.B = np.zeros((self.FilterCount, 1)).astype('float32')
        #self.SaveInitialValue()

    def Rotate180(self):
        self.WT = np.zeros(self.W.shape).astype(np.float32)
        for i in range(self.FilterCount):
            for j in range(self.KernalCount):
                self.WT[i,j] = np.rot90(self.W[i,j], 2)
        return self.WT

    def ClearGrads(self):
        self.dW = np.zeros(self.W.shape).astype(np.float32)
        self.dB = np.zeros(self.B.shape).astype(np.float32)

    def MeanGrads(self, m):
        self.dW = self.dW / m
        self.dB = self.dB / m

    @staticmethod
    def InitialConvParameters(shape, method):
        assert(len(shape) == 4)
        num_input = shape[2]
        num_output = shape[3]
        
        if method == InitialMethod.Zero:
            W = np.zeros(shape).astype('float32')
        elif method == InitialMethod.Normal:
            W = np.random.normal(shape).astype('float32')
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2/num_input*num_output), shape).astype('float32')
        elif method == InitialMethod.Xavier:
            t = math.sqrt(6/(num_output+num_input))
            W = np.random.uniform(-t, t, shape).astype('float32')
        return W

