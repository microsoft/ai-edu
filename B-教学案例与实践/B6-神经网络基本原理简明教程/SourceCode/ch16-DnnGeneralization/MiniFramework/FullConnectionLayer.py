# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from MiniFramework.Layer import *
from MiniFramework.WeightsBias import *
from MiniFramework.Parameters import *

class FcLayer(CLayer):
    def __init__(self, input_size, output_size, param):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = WeightsBias(self.input_size, self.output_size, param.init_method, param.optimizer, param.eta)
        self.weights.InitializeWeights()
        self.regular = param.regular
        self.lambd = param.lambd

    def forward(self, input, train=True):
        self.input_shape = input.shape
        if input.ndim == 3: # come from pooling layer
            self.x = input.reshape(input.size, 1)
        else:
            self.x = input
        self.z = np.dot(self.weights.W, self.x) + self.weights.B
        return self.z

    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, idx):
        dZ = delta_in
        m = self.x.shape[1]
        if self.regular == RegularMethod.L2:
            self.weights.dW = (np.dot(dZ, self.x.T) + self.lambd * self.weights.W) / m
        elif self.regular == RegularMethod.L1:
            self.weights.dW = (np.dot(dZ, self.x.T) + self.lambd * np.sign(self.weights.W)) / m
        else:
            self.weights.dW = np.dot(dZ, self.x.T) / m
        # end if
        self.weights.dB = np.sum(dZ, axis=1, keepdims=True) / m
        # calculate delta_out for lower level
        if idx == 0:
            return None
        
        delta_out = np.dot(self.weights.W.T, dZ)

        if len(self.input_shape) > 2:
            return delta_out.reshape(self.input_shape)
        else:
            return delta_out

    def pre_update(self):
        self.weights.pre_Update()

    def update(self):
        self.weights.Update()
        
    def save_parameters(self, name):
        self.weights.SaveResultValue(name)

    def load_parameters(self, name):
        self.weights.LoadResultValue(name)
