# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from MiniFramework.Layer import *
from MiniFramework.WeightsBias_1_0 import *
from MiniFramework.HyperParameters_4_0 import *

class FcLayer_1_0(CLayer):
    def __init__(self, input_size, output_size, param):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = WeightsBias_1_0(self.input_size, self.output_size, param.init_method, param.eta)

    def initialize(self, folder):
        self.weights.InitializeWeights(folder, False)

    def forward(self, input, train=True):
        self.input_shape = input.shape
        self.x = input
        self.z = np.dot(self.x, self.weights.W) + self.weights.B
        return self.z

    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, layer_idx):
        dZ = delta_in
        m = self.x.shape[0]
        self.weights.dW = np.dot(self.x.T, dZ) / m
        # end if
        self.weights.dB = np.sum(dZ, axis=0, keepdims=True) / m
        # calculate delta_out for lower level
        if layer_idx == 0:
            return None
        
        #delta_out = np.dot(self.weights.W.T, dZ)
        delta_out = np.dot(dZ, self.weights.W.T)

        if len(self.input_shape) > 2:
            return delta_out.reshape(self.input_shape)
        else:
            return delta_out

    def pre_update(self):
        self.weights.pre_Update()

    def update(self):
        self.weights.Update()
        
    def save_parameters(self, folder, name):
        self.weights.SaveResultValue(folder, name)

    def load_parameters(self, folder, name):
        self.weights.LoadResultValue(folder, name)
