# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

class CTanh(object):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return da, dz

class CSigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return da, dz

