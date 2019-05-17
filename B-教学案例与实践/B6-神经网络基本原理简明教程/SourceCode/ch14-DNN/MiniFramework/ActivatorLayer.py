# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from MiniFramework.Layer import *

# sigmoid and softmax
class ClassificationLayer(CLayer):
    def __init__(self, activator):
        self.activator = activator

    def forward(self, input, train=True):
        self.input_shape = input.shape
        self.x = input
        self.a = self.activator.forward(self.x)
        return self.a

    # 对分类函数的求导已经和损失函数合并计算了，所以不需要再做，直接回传误差给上一层
    def backward(self, delta_in, flag):
        dZ = delta_in
        return dZ

class ActivatorLayer(CLayer):
    def __init__(self, activator):
        self.activator = activator

    def forward(self, input, train=True):
        self.input_shape = input.shape
        self.x = input
        self.a = self.activator.forward(self.x)
        return self.a

    # 把激活函数算做是当前层，上一层的误差传入后，先经过激活函数的导数，而得到本层的针对z值的误差
    def backward(self, delta_in, flag):
        dZ,_ = self.activator.backward(self.x, self.a, delta_in)
        return dZ

class CActivator(object):
    # z = 本层的wx+b计算值矩阵
    def forward(self, z):
        pass

    # z = 本层的wx+b计算值矩阵
    # a = 本层的激活函数输出值矩阵
    # delta = 上（后）层反传回来的梯度值矩阵
    def backward(self, z, a, delta):
        pass

    def get_name(self):
        return self.__class__.__name__

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


class Softmax(CActivator):
    def forward(self, z):
        shift_z = z - np.max(z, axis=0)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=0)
        return a

    # softmax层，相当于最后一层，直接A-Y，不需要做反向求导
    def backward(self, z, a, delta_in):
        return delta_in, None

