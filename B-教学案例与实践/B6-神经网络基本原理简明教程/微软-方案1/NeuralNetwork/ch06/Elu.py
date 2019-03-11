# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

import DrawCurve

class CElu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, z):
        return np.array([x if x > 0 else self.alpha * (np.exp(x) - 1) for x in z])


    def backward(self, z, a, delta):
        da = np.array([1 if x > 0 else self.alpha * np.exp(x) for x in z])
        dz = np.multiply(delta, da)
        return da, dz

if __name__ == '__main__':
    DrawCurve.Draw(-4,4,CElu(0.8),"ELU Function","Derivative of ELU")
