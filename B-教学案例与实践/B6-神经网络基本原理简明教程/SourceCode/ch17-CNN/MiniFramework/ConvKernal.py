# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import math
from MiniFramework.EnumDef_6_0 import *
from MiniFramework.WeightsBias_2_1 import *
from MiniFramework.Optimizer_1_0 import *

"""
Weights and Bias: 一个Weights可以包含多个卷积核Kernal，一个卷积核可以包含多个过滤器Filter
WK - Kernal 卷积核数量(等于输出通道数量), 每个WK有一个Bias
WC - Channel 输入通道数量
FH - Filter Height
FW - Filter Width
"""
class ConvKernal(WeightsBias_2_1):
    def __init__(self, output_c, input_c, filter_h, filter_w, init_method, optimizer_name, eta):
        self.KernalCount = output_c
        self.FilterCount = input_c
        self.FilterHeight = filter_h
        self.FilterWidth = filter_w
        self.init_method = init_method
        self.optimizer_name = optimizer_name
        self.learning_rate = eta
        self.KernalShape = (self.KernalCount, self.FilterCount, self.FilterHeight, self.FilterWidth)

    def Initialize(self, folder, name, create_new):
        self.init_file_name = str.format(
            "{0}/{1}_{2}_{3}_{4}_{5}_init.npz", 
            folder, name, self.KernalCount, self.FilterCount, self.FilterHeight, self.FilterWidth)
        self.result_file_name = str.format("{0}/{1}_result.npz", folder, name)

        if create_new:
            self.CreateNew()
        else:
            self.LoadExistingParameters()

        # end if
        self.CreateOptimizers()

        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def CreateNew(self):
        self.W = ConvKernal.InitialConvParameters(self.KernalShape, self.init_method)
        self.B = np.zeros((self.KernalCount, 1))
        #self.SaveInitialValue()

    def Rotate180(self):
        self.WT = np.zeros(self.W.shape).astype(np.float32)
        for i in range(self.KernalCount):
            for j in range(self.FilterCount):
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
            W = np.zeros(shape)
        elif method == InitialMethod.Normal:
            W = np.random.normal(shape)
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2/num_input*num_output), shape)
        elif method == InitialMethod.Xavier:
            t = math.sqrt(6/(num_output+num_input))
            W = np.random.uniform(-t, t, shape)
        return W

