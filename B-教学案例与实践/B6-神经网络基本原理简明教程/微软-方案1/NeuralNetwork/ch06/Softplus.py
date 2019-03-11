# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

import DrawCurve

class CSoftplus(object):
    def forward(self, z):
        a = np.log(1 + np.exp(z))
        return a

    def backward(self, z, a, delta):
        p = np.exp(z) 
        da = p / (1 + p)
        dz = np.multiply(delta, da)
        return da, dz


if __name__ == '__main__':
    DrawCurve.Draw(-5,5,CSoftplus(),"Softplus Function","Derivative of Softplus")
