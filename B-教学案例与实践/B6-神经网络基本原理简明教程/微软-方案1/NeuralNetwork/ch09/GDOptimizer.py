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
        vt_new = self.alpha * self.vt + self.eta * grad
        theta = theta - vt_new
        self.vt = vt_new
        return theta

class AdaGrad(GDOptimizer):
    def __init__(self, eta):
        self.delta = 1e-7
        self.eta = eta
        self.r = 0

    def step(self, theta, grad):
        self.r = self.r + np.multiply(grad, grad)
        alpha = self.eta / (np.sqrt(self.r) + self.delta)
        theta = theta - alpha * grad
        return theta
  
class CRMSProp(GDOptimizer):
    def __init__(self, eta):
        self.eta = eta
        self.p = 0.9
        self.delta = 1e-6
        self.r = 0

    def step(self, theta, grad):
        grad2 = np.multiply(grad, grad)
        self.r = self.p * self.r + (1-self.p) * grad2
        alpha = self.eta / np.sqrt(self.delta + self.r)
        theta = theta - np.multiply(alpha, grad)
        return theta