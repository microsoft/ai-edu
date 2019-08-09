# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *
from MiniFramework.jit_utility import *

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
        
    def initialize(self, folder, name):
        self.init_file_name = str.format("{0}/{1}_init.npz", folder, name)

    def forward(self, x, train=True):
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

    def save_parameters(self):
        np.save(self.init_file_name, self.pooling_type)

    def load_parameters(self):
        self.mode = np.load(self.init_file_name)
