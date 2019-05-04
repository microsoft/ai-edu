# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

class CActivator(object):
    # z = wx+b
    def forward(self, z):
        pass

    # z = wx+b
    # a = a(z)
    # delta = delta(error) from upper level
    def backward(self, z, a, delta):
        pass


# no activation
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

    # check if z >0, not a
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

