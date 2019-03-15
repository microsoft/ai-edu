# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

class CMomentum(object):
    def __init__(self, eta):
        self.vt_1 = 0
        self.eta = eta
        self.gamma = 0.9

    def step(self, theta, grad):
        vt = self.gamma * self.vt_1 + self.eta * grad
        theta = theta - vt
        self.vt_1 = vt
        return theta

