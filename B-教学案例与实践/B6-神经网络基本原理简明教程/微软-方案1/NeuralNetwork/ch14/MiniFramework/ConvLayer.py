# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np
import timeit

from MiniFramework.Layer import *
from MiniFramework.Activators import *
from MiniFramework.ConvWeightsBias import *
from MiniFramework.Parameters import *

from MiniFramework.jit_utility import *

class ConvLayer(CLayer):
    # define the number of input and output channel, also the filter size
    def __init__(self, input_c, output_c, filter_h, filter_w, input_h, input_w, stride, padding, activator):
        self.num_input_channel = input_c
        self.num_output_channel = output_c
        self.filter_height = filter_h
        self.filter_width = filter_w
        self.input_height = input_h
        self.input_width = input_w
        self.stride = stride
        self.padding = padding
        self.activator = activator

        self.WeightsBias = ConvWeightsBias(self.num_output_channel, self.num_input_channel, self.filter_height, self.filter_width)
        self.output_height, self.output_width = calculate_output_size(
            self.input_height, self.input_width, 
            self.filter_height, self.filter_width, 
            self.padding, self.stride)

    """
    输入数据
    N：样本图片数量（比如一次计算10张图片）
    C：图片通道数量（比如红绿蓝三通道）
    H：图片高度（比如224）
    W：图片宽度（比如224）
    思维卷积操作
    """
    
    def forward(self, x):
        assert(x.ndim == 4)
        self.x = x
        assert(self.x.shape[1] == self.num_input_channel)
        assert(self.x.shape[2] == self.input_height)
        assert(self.x.shape[3] == self.input_width)
        self.batch_size = self.x.shape[0]

        if self.padding > 0:
            self.padded = np.pad(self.x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 'constant',  constant_values=(0,0))
        else:
            self.padded = self.x
        #end if

        #self.z = np.zeros((self.input_shape[0], self.num_output_channel, self.output_height, self.output_width)
        self.z = jit_conv_4d(self.padded, self.WeightsBias.W, self.WeightsBias.B, self.output_height, self.output_width, self.stride)
        self.a = self.activator.forward(self.z)
        return self.z, self.a

    def forward_fast(self, x):
        FN, C, FH, FW = self.WeightsBias.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.padding - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.padding - FW) / self.stride)
        col_x = im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.WeightsBias.W.reshape(FN, -1).T
        out = np.dot(col_x, col_W) + self.WeightsBias.B.reshape(-1,FN)
        self.z = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) 
        self.x = x
        self.col_x = col_x
        self.col_W = col_W
        self.a = self.activator.forward(self.z)
        return self.z, self.a

    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, flag):
        assert(delta_in.ndim == 4)
        assert(delta_in.shape == self.a.shape)
        
        # 计算激活函数的导数
        dz,_ = self.activator.backward(self.z, self.a, delta_in)
        
        # 转换误差矩阵尺寸
        dz_padded, dz_stride_1 = expand_delta_map(dz, self.batch_size, self.num_input_channel, self.input_height, self.input_width, self.output_height, self.output_width, self.filter_height, self.filter_width, self.padding, self.stride)
        
        # 计算本层的权重矩阵的梯度
        self._calculate_weightsbias_grad(dz_stride_1)

        # 计算本层输出到下一层的误差矩阵
        delta_out = self._calculate_delta_out(dz_padded)
        return delta_out

    # 用输入数据乘以回传入的误差矩阵,得到卷积核的梯度矩阵
    def _calculate_weightsbias_grad(dz):
        self.WeightsBias.ClearGrads()
        pad_h, pad_w = calculate_padding_size(self.input_height, self.input_width, dz.shape[2], dz.shape[3], self.filter_height, self.filter_width, 1)
        input_padded = np.pad(self.x, ((0,0),(0,0),(pad_h, pad_h),(pad_w,pad_w)))
        for bs in range(batch_size):
            for oc in range(self.num_output_channel):   # == kernal count
                for ic in range(self.num_input_channel):    # == filter count
                    w_grad = np.zeros(self.filter_height, self.filter_width)
                    conv2d(input_padded[bs,ic], dz[bs,oc], 0, w_grad)
                    self.WeightsBias.W_grad[oc,ic] += w_grad
                #end ic
                self.WeightsBias.B_grad[oc] += dz[bs,oc].sum()
            #end oc
        #end bs
        self.WeightsBias.MeanGrads(batch_size)

        
    # 用输入误差矩阵乘以（旋转180度后的）卷积核
    def _calculate_delta_out(dz, flag):
        delta_out = np.zeros(self.x.shape)
        if flag != LayerIndexFlags.FirstLayer:
            rot_weights = self.WeightsBias.Rotate180()
            for bs in range(batch_size):
                for oc in range(self.num_output_channel):    # == kernal count
                    delta_per_input = np.zeros((self.input_height, self.input_width))
                    for ic in range(self.num_input_channel): # == filter count
                        conv2d(dz[bs,oc], rot_weights[oc,ic], 0, delta_per_input)
                        delta_out[bs,ic] += delta_per_input
                    #END IC
                #end oc
            #end bs
        # end if
        return delta_out

    def pre_update(self):
        self.weights.pre_Update()

    def update(self):
        self.weights.Update()
        
    def save_parameters(self, name):
        self.weights.SaveResultValue(name)

    def load_parameters(self, name):
        self.weights.LoadResultValue(name)

#end class

def conv1():
    r1,_ = cl.forward(x)
    return r1

def conv2():
    r2,_ = cl.forward_fast(x)
    return r2


if __name__ == '__main__':
    # 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
    x = np.random.randn(64, 3, 28, 28)
    cl = ConvLayer(3, 10, 5, 5, 28, 28, 1, 0, Relu())
    cl.Initialize()


    r1 = conv1()
    r2 = conv2()

    print(np.allclose(r1, r2))
    print(r1.sum(), r2.sum())

    num=10
    t4 = timeit.timeit('conv1()','from __main__ import conv1', number=num)
    print("t4:", t4, t4/num)

    t5 = timeit.timeit('conv2()','from __main__ import conv2', number=num)
    print("t5:", t5, t5/num)



