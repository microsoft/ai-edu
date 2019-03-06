# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

import DrawCurve

class CTanh(object):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return da, dz

if __name__ == '__main__':
    DrawCurve.Draw(-5,5,CTanh(),"Tanh Function","Derivative of Tanh")
