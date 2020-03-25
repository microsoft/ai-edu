# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

class CSoftplus(object):
    def forward(self, z):
        a = np.log(1 + np.exp(z))
        return a

    def backward(self, z, a, delta):
        p = np.exp(z) 
        da = p / (1 + p)
        dz = np.multiply(delta, da)
        return da, dz
