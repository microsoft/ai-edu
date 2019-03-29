# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


import numpy as np
from enum import Enum

from Layer import *
from FCLayer import *

class CNet(object):
    def __init__(self, optimizer_type, loss_function):
        self.optimizer_type = optimizer_type
        self.loss_function = loss_function
        self.layer_list = []
        self.layer_name = []
        self.output = np.zeros((1,1))
        self.layer_count = 0

    def add_layer(self, layer, name=""):
        self.layer_list.append(layer)
        self.layer_name.append(name)
        self.layer_count += 1

    def forward(self, input_array):
        input = input_array
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            output = layer.forward(input)
            input = output

        self.output = output
        return self.output

    def backward(self, X, Y):
        delta_in = self.output - Y
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            flag = self.get_layer_index(i)
            delta_out = layer.backward(delta_in, flag)
            # move back to previous layer
            delta_in = delta_out

    def update(self, learning_rate):
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            layer.update(learning_rate)

    def get_layer_index(self, idx):
        if self.layer_count == 1:
            return LayerIndexFlags.SingleLayer
        else:
            if idx == self.layer_count - 1:
                return LayerIndexFlags.LastLayer
            elif idx == 0:
                return LayerIndexFlags.FirstLayer
            else:
                return LayerIndexFlags.MiddleLayer

    def train(self, X, Y, max_iteration, learning_rate):
        if X.ndim == 2:
            num_feature = X.shape[0]
            num_example = X.shape[1]
        elif X.ndim == 4:
            num_example = X.shape[0]
            num_feature = X.shape[1] * X.shape[2] * X.shape[3]

        num_output = Y.shape[0]

       # num_example = 2000

        for i in range(max_iteration):
            print(i)
            for j in range(num_example):
                if j%10==0:
                    print(i, ":", j)
                if X.ndim == 2:
                    x = X[:,j].reshape(num_feature, 1)
                elif X.ndim == 4:
                    x = X[j]    # x.ndim == 3
                y = Y[:,j].reshape(num_output, 1)
                self.forward(x)
                self.backward(x, y)
                self.update(learning_rate)

    def inference(self, X):
        self.forward(X)
        return self.output

    def save_parameters(self):
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            name = self.layer_name[i]
            layer.save_parameters(name)

    def load_parameters(self):
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            name = self.layer_name[i]
            layer.load_parameters(name)
