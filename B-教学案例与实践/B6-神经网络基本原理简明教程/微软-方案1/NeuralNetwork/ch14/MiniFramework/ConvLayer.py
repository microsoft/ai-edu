# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np

from MiniFramework.Layer import *
from MiniFramework.Activators import *
from MiniFramework.ConvWeightsBias import *
from MiniFramework.Parameters import *

class ConvLayer(CLayer):
    def __init__(self, num_input_channel, num_output_channel, num_filter_size, stride, padding, activator):
        self.num_input_channel = num_input_channel
        self.num_output_channel = num_output_channel
        self.num_filter_size = num_filter_size
        self.stride = stride
        self.padding = padding
        self.activator = activator

    def Initialize(self):
        self.weights = ConvWeightsBias(self.num_output_channel, self.num_input_channel, self.num_filter_size, self.num_filter_size)
        self.weights.Initialize();

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
