# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

import DrawCurve

class CLeakyRelu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * x for x in z])

    def backward(self, a, delta):
        da = np.array([1 if x > 0 else self.alpha for x in a])
        dz = 0
        return da, dz

if __name__ == '__main__':
    DrawCurve.Draw(-5,5,CLeakyRelu(0.01),"Leaky Relu Function","Derivative of Leaky Relu")
