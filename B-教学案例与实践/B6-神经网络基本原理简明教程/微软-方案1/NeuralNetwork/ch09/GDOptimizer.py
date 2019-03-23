# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

class CGDOptimizer(object):
    def __init__(self):
        Pass

    def update(self, theta, grad):
        Pass

class SGD(CGDOptimizer):
    def __init__(self, eta):
        self.eta = eta

    def update(self, theta, grad):
        theta = theta - self.eta * grad
        return theta


class Momentum(object):
    def __init__(self, eta):
        self.vt = 0
        self.eta = eta
        self.alpha = 0.9

    def step(self, theta, grad):
        vt_new = self.alpha * self.vt - self.eta * grad
        self.vt = vt_new
        theta = theta + vt_new
        return theta
