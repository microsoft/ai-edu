# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np
from enum import Enum


from MiniFramework.Layer import *
from MiniFramework.Activators import *
from MiniFramework.ConvWeightsBias import *
from MiniFramework.Parameters import *
from MiniFramework.jit_utility import *


class PoolingTypes(Enum):
    MAX = 0,
    MEAN = 1,


class PoolingLayer(CLayer):
    def __init__(self,
                input_shape,    # (input_c, input_h, input_w)
                pool_shape,     # (pool_h, pool_w)
                stride, 
                pooling_type):  # MAX, MEAN
        self.num_input_channel = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.pool_height = pool_shape[0]
        self.pool_width = pool_shape[1]
        self.stride = stride
        self.pooling_type = pooling_type

        self.pool_size = self.pool_height * self.pool_width
        self.output_height = (self.input_height - self.pool_height) // self.stride + 1
        self.output_width = (self.input_width - self.pool_width) // self.stride + 1
        self.output_shape = (self.num_input_channel, self.output_height, self.output_width)
        self.output_size = self.num_input_channel * self.output_height * self.output_width
        
    def forward(self, x):
        assert(x.ndim == 4)
        self.x = x
        self.batch_size = self.x.shape[0]
        self.z = max_pool_forward(self.x, self.batch_size, self.num_input_channel, self.output_height, self.output_width, self.pool_height, self.pool_width, self.stride)
        return self.z

    def backward(self, delta_in, flag):
        assert(delta_in.ndim == 4)
        assert(delta_in.shape == self.z.shape)

        delta_out = max_pool_backward(self.x, delta_in, self.batch_size, self.num_input_channel, self.output_height, self.output_width, self.pool_height, self.pool_width, self.stride)
        return delta_out

        """
        delta_out = np.zeros(self.x.shape)
     
        for b in range(self.batch_size):
            for c in range(self.num_input_channel):
                for i in range(self.output_height):
                    i_start = i * self.stride
                    i_end = i_start + self.pool_height
                    for j in range(self.output_width):
                        j_start = j * self.stride
                        j_end = j_start + self.pool_width
                        if self.pooling_type == PoolingTypes.MAX:
                            m,n = PoolingLayer.get_max_index(self.x[b,c], i_start, i_end, j_start, j_end)
                            delta_out[b,c,m,n] = delta_in[b,c,i,j]
                        else: 
                            delta_out[b,c,i_start:i_end, j_start:j_end] = delta_in[b,c,i,j] / self.pool_size
        """
        

    def save_parameters(self, name):
        np.save(name + "_type", self.pooling_type)

    def load_parameters(self, name):
        self.mode = np.load(name+"_type.npy")
    
    @staticmethod
    def get_max_index(input, i_start, i_end, j_start, j_end):
        assert(input.ndim == 2)
        max_i = i_start
        max_j = j_start
        max_value = input[i_start,j_start]
        for i in range(i_start,i_end):
            for j in range(j_start,j_end):
                if input[i,j] > max_value:
                    max_value = input[i,j]
                    max_i, max_j = i, j

        return max_i, max_j

