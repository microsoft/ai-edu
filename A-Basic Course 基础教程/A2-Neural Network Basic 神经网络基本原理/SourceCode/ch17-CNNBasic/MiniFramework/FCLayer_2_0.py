# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *
from MiniFramework.WeightsBias_2_1 import *
from MiniFramework.HyperParameters_4_2 import *

class FcLayer_2_0(CLayer):
    def __init__(self, input_size, output_size, hp):
        self.input_size = input_size
        self.output_size = output_size
        self.wb = WeightsBias_2_1(self.input_size, self.output_size, hp.init_method, hp.optimizer_name, hp.eta)
        self.regular_name = hp.regular_name
        self.regular_value = hp.regular_value

    def initialize(self, folder, name):
        self.wb.Initialize(folder, name, False)

    def forward(self, input, train=True):
        self.input_shape = input.shape
        if input.ndim == 4: # come from pooling layer
            self.x = input.reshape(self.input_shape[0],-1)
        else:
            self.x = input
        self.z = np.dot(self.x, self.wb.W) + self.wb.B
        return self.z

    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, layer_idx):
        dZ = delta_in
        m = self.x.shape[0]
        if self.regular_name == RegularMethod.L2:
            self.wb.dW = (np.dot(self.x.T, dZ) + self.regular_value * self.wb.W) / m
        elif self.regular_name == RegularMethod.L1:
            self.wb.dW = (np.dot(self.x.T, dZ) + self.regular_value * np.sign(self.wb.W)) / m
        else:
            self.wb.dW = np.dot(self.x.T, dZ) / m
        # end if
        self.wb.dB = np.sum(dZ, axis=0, keepdims=True) / m
        # calculate delta_out for lower level
        if layer_idx == 0:
            return None
        
        delta_out = np.dot(dZ, self.wb.W.T)

        if len(self.input_shape) > 2:
            return delta_out.reshape(self.input_shape)
        else:
            return delta_out

    def pre_update(self):
        self.wb.pre_Update()

    def update(self):
        self.wb.Update()
        
    def save_parameters(self):
        self.wb.SaveResultValue()

    def load_parameters(self):
        self.wb.LoadResultValue()
