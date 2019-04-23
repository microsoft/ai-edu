# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

class CActivator(object):
    # z = 本层的wx+b计算值矩阵
    def forward(self, z):
        return z

    # z = 本层的wx+b计算值矩阵
    # a = 本层的激活函数输出值矩阵
    # delta = 上（后）层反传回来的梯度值矩阵
    def backward(self, z, a, delta):
        # da是本激活函数的导数值
        da = 1
        # dz是激活函数导数值与传回误差项相计算的结果
        dz = delta
        return dz, da


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

# z为矩阵
class Relu(CActivator):
    def __init__(self):
        self.mem_mask = None

    def forward(self, z):
        # 简单实现
        #a = np.maximum(z, 0)
        #return a

        # 上下文实现
        # 记录前向的时小于0的位置信息在mask里面
        self.mem_mask = (z <= 0)
        a = z.copy()
        a[self.mem_mask] = 0
        return a

    # 注意relu函数判断是否大于1的根据是正向的wx+b=z的值，而不是a值
    def backward(self, z, a, delta):
        # 简单实现
        #da = np.zeros(z.shape)
        #da[z>0] = 1
        #dz = da * delta
        #return dz, da

        # 上下文实现
        delta[self.mem_mask] = 0    # 直接把正向时小于0的位置置零
        dz = delta
        return dz, None # 此处da可以返回self.mem_mask


class Softmax(CActivator):
    def forward(self, z):
        shift_z = z - np.max(z, axis=0)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=0)
        return a

