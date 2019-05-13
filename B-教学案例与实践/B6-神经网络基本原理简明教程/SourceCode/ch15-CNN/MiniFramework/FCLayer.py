# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np

from MiniFramework.Layer import *
from MiniFramework.Activators import *
from MiniFramework.WeightsBias import *
from MiniFramework.Parameters import *

class FcLayer(CLayer):
    def __init__(self, input_size, output_size, activator, param):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.weights = WeightsBias(self.input_size, self.output_size, param.init_method, param.optimizer_name, param.eta)
        self.weights.Initialize()

    def forward(self, input):
        self.input_shape = input.shape
        batch_size = input.shape[0]
        if input.ndim == 4: # come from pooling layer
            self.x = input.reshape(batch_size,-1).T
        else:
            self.x = input
        self.z = np.dot(self.weights.W, self.x) + self.weights.B
        self.a = self.activator.forward(self.z)
        return self.a

    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, flag):
        if flag == LayerIndexFlags.LastLayer or flag == LayerIndexFlags.SingleLayer:
            dZ = delta_in
        else:
            #dZ = delta_in * self.activator.backward(self.a)
            dZ,_ = self.activator.backward(self.z, self.a, delta_in)

        m = self.x.shape[1]
        self.weights.dW = np.dot(dZ, self.x.T) / m
        self.weights.dB = np.sum(dZ, axis=1, keepdims=True) / m
        # calculate delta_out for lower level
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
