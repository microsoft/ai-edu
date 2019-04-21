# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np

"""
Weights and Bias: 一个Weights可以包含多个卷积核Kernal，一个卷积核可以包含多个过滤器Filter
WK - Kernal 卷积核数量(等于输出通道数量), 每个WK有一个Bias
WC - Channel 输入通道数量
FH - Filter Height
FW - Filter Width
"""
class ConvWeightsBias(object):
    def __init__(self, output_c, input_c, filter_h, filter_w):
        self.KernalCount = output_c
        self.FilterCount = input_c
        self.FilterHeight = filter_h
        self.FilterWidth = filter_w

        self.W = np.zeros((self.KernalCount, self.FilterCount, self.FilterHeight, self.FilterWidth))
        #self.W = np.random.randn(self.KernalCount, self.FilterCount, self.FH, self.FW)
        self.B = np.zeros((self.KernalCount, 1))# + 0.1

        self.W_grad = np.zeros(self.W.shape)
        self.B_grad = np.zeros(self.B.shape)

    def Rotate180(self):
        self.WT = np.zeros(self.W.shape)
        for i in range(self.KernalCount):
            for j in range(self.FilterCount):
                self.WT[i,j] = np.rot90(self.W[i,j], 2)
        return self.WT

    def ClearGrads(self):
        self.W_grad = np.zeros(self.W.shape)
        self.B_grad = np.zeros(self.B.shape)

    def MeanGrads(self, m):
        self.W_grad = self.W_grad / m
        self.B_grad = self.B_grad / m


if __name__ == '__main__':
    wb = ConvWeightsBias(4,2,3,3)

