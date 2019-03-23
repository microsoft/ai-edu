# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

class GDOptimizer(object):
    def __init__(self):
        Pass

    def update(self, theta, grad):
        Pass

class SGD(GDOptimizer):
    def __init__(self, eta):
        self.eta = eta

    def update(self, theta, grad):
        theta = theta - self.eta * grad
        return theta


class Momentum(GDOptimizer):
    def __init__(self, eta):
        self.vt = 0
        self.eta = eta
        self.alpha = 0.9

    def update(self, theta, grad):
        vt_new = self.eta * grad - self.alpha * self.vt
        theta = theta - vt_new
        self.vt = vt_new
        return theta

    