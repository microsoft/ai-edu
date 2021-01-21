# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 1.1
what's new:
- add softmax
- remove tanh
"""

import numpy as np

class Logistic(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

class Softmax(object):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a

if __name__ == '__main__':
    z = np.array([[3,1,-3],[1,-3,3]]).reshape(2,3)
    a = Softmax().forward(z)
    print(a)