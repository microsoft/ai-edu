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
    def __init__(self, num_Kernal, num_Channel, num_Height, num_Width):
        self.WK = num_Kernal
        self.WC = num_Channel
        self.FH = num_Height
        self.FW = num_Width

    def Initialize():
        self.W = np.zeros((self.WK, self.WC, self.FH, self.FW))
        self.B = np.zeros((self.WK, 1))


if __name__ == '__main__':
    wb = ConvWeightsBias(2,3,5,5)
