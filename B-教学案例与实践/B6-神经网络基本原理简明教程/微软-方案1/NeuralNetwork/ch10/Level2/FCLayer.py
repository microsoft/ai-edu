# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from Layer import *
from Activators import *

class FcLayer(CLayer):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = init_array((output_size, input_size), "norm")
        self.bias = np.zeros((output_size, 1))
        self.activator = activator
        #self.x = np.zeros()    # input from lower layer
        #self.z = np.zeros()    # weights multiply for current layer
        #self.a = np.zeros()    # outpu to upper layer

    def forward(self, input):
        self.input_shape = input.shape
        if input.ndim == 3: # come from pooling layer
            self.x = input.reshape(input.size, 1)
        else:
            self.x = input
        self.z = np.dot(self.weights, self.x) + self.bias
        self.a = self.activator.forward(self.z)
        return self.a

    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, flag):
        if flag == LayerIndexFlags.LastLayer or flag == LayerIndexFlags.SingleLayer:
            dZ = delta_in
        else:
            #dZ = delta_in * self.activator.backward(self.a)
            dZ = self.activator.backward(self.a, delta_in)

        delta_out = np.dot(self.weights.T, dZ)
        self.dW = np.dot(dZ, self.x.T)
        self.dB = np.sum(dZ, axis=1, keepdims=True)

        if len(self.input_shape) > 2:
            return delta_out.reshape(self.input_shape)
        else:
            return delta_out

    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dW
        self.bias = self.bias - learning_rate * self.dB
        
    def save_parameters(self, name):
        np.save(name+"_w", self.weights)
        np.save(name+"_b", self.bias)

    def load_parameters(self, name):
        self.weights = np.load(name+"_w.npy")
        self.bias = np.load(name+"_b.npy")
