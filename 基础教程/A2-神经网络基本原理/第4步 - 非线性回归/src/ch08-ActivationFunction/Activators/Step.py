# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

class CStep(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def forward(self, z):
        a = np.array([1 if x > self.threshold else 0 for x in z])
        return a

    def backward(self, z, a, delta):
        da = np.zeros(a.shape)
        dz = da
        return da, dz
