# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np

from MiniFramework.Layer import *
from MiniFramework.Activators import *
from MiniFramework.ConvWeightsBias import *
from MiniFramework.Parameters import *

class ConvLayer(CLayer):
    # define the number of input and output channel, also the filter size
    def __init__(self, input_n, output_n, filter_h, filter_w, input_h, input_w, stride, padding, activator):
        self.num_input_channel = input_n
        self.num_output_channel = output_n
        self.filter_height = filter_h
        self.filter_width = filter_w
        self.input_height = input_h
        self.input_width = input_w
        self.stride = stride
        self.padding = padding
        self.activator = activator

    def Initialize(self):
        self.W = ConvWeightsBias(self.num_output_channel, self.num_input_channel, self.filter_height, self.filter_width)
        self.W.Initialize();
        
        self.output_height, self.output_width = calculate_output_size(self.input_height, self.input_width, self.num_filter_size, self.num_filter_size, self.padding, self.stride)
        self.output_shape = (self.num_output_channel, self.output_height, self.output_width)
        
        # output of conv
        self.z = np.zeros(self.output_shape)
        # output of activator
        self.a = np.zeros(self.output_shape)


    """
    输入数据
    N：样本图片数量（比如一次计算10张图片）
    C：图片通道数量（比如红绿蓝三通道）
    H：图片高度（比如224）
    W：图片宽度（比如224）
    思维卷积操作
    """
    
    def forward(self, x):
        self.input_shape = x.shape
        assert(x.ndim == 4)
        self.x = x

        if self.padding > 0:
            self.padded = np.pad(self.x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 'constant',  constant_values=(0,0))
        else:
            self.padded = self.x
        #end if



    def forward_fast(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        col_x = im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col_x, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        self.col_x = col_x
        self.col_W = col_W
        return out        



    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, flag):


    def pre_update(self):
        self.weights.pre_Update()

    def update(self):
        self.weights.Update()
        
    def save_parameters(self, name):
        self.weights.SaveResultValue(name)

    def load_parameters(self, name):
        self.weights.LoadResultValue(name)

    def im2col(self):
