# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 2.0
"""

import numpy as np

class CActivator(object):
    # z = 本层的wx+b计算值矩阵
    def forward(self, z):
        pass

    # z = 本层的wx+b计算值矩阵
    # a = 本层的激活函数输出值矩阵
    # delta = 上（后）层反传回来的梯度值矩阵
    def backward(self, z, a, delta):
        pass


# 直传函数，相当于无激活
class Identity(CActivator):
    def forward(self, z):
        return z

    def backward(self, z, a, delta):
        return delta, a


class Sigmoid(CActivator):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return dz, da


class Tanh(CActivator):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return dz, da


class Relu(CActivator):
    def forward(self, z):
        a = np.maximum(z, 0)
        return a

    # 注意relu函数判断是否大于1的根据是正向的wx+b=z的值，而不是a值
    def backward(self, z, a, delta):
        da = np.zeros(z.shape)
        da[z>0] = 1
        dz = da * delta
        return dz, da
