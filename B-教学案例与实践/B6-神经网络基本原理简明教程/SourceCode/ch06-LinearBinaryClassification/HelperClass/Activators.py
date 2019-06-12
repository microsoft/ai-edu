# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

class Sigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

