# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

import DrawCurve

class CSigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return da, dz


if __name__ == '__main__':
    DrawCurve.Draw(-7,7,CSigmoid(),"Sigmoid Function","Derivative of Sigmoid")
