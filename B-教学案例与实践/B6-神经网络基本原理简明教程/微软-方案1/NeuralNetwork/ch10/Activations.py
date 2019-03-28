# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

class CSigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return da, dz

class CSoftmax(object):
    def forward(self, Z):
        shift_z = Z - np.max(Z, axis=0)
        exp_z = np.exp(shift_z)
        A = exp_z / np.sum(exp_z, axis=0)
        return A

