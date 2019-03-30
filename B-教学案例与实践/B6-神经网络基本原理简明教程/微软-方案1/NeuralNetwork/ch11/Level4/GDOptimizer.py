# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from enum import Enum

class OptimizerName(Enum):
    SGD = 0,
    Momentum = 1,
    Nag = 2,
    AdaGrad = 3,
    AdaDelta = 4,
    RMSProp = 5,
    Adam = 6


class GDOptimizerFactory(object):
    @staticmethod
    def CreateOptimizer(eta, name = OptimizerName.SGD):
        if name == OptimizerName.SGD:
            optimizer = SGD(eta)
        elif name == OptimizerName.Adam:
            optimizer = Adam(eta)
        elif name == OptimizerName.AdaGrad:
            optimizer = AdaGrad(eta)
        elif name == OptimizerName.Momentum:
            optimizer = Momentum(eta)
        elif name == OptimizerName.Nag:
            optimizer = Nag(eta)
        elif name == OptimizerName.RMSProp:
            optimizer = RMSProp(eta)
        elif name == OptimizerName.AdaDelta:
            optimizer = AdaDelta(eta)

        return optimizer

class GDOptimizer(object):
    def __init__(self):
        pass

    def pre_update(self, theta):
        pass 

    def update(self, theta, grad):
        pass

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
        self.eps = 1e-6
        self.eta = eta
        self.r = 0

    def update(self, theta, grad):
        self.r = self.r + np.multiply(grad, grad)
        alpha = self.eta / (self.eps + np.sqrt(self.r))
        theta = theta - alpha * grad
        return theta

class AdaDelta(GDOptimizer):
    def __init__(self, eta):
        self.eps = 1e-5
        self.r = 0
        self.s = 0
        self.alpha = 0.9

    def update(self, theta, grad):
        grad2 = np.multiply(grad, grad)
        self.s = self.alpha * self.s + (1-self.alpha)*grad2
        d_theta = np.sqrt((self.eps + self.r)/(self.eps + self.s)) * grad
        theta = theta - d_theta
        d_theta2 = np.multiply(d_theta, d_theta)
        self.r = self.alpha * self.r + (1-self.alpha) * d_theta2
        return theta

class RMSProp(GDOptimizer):
    def __init__(self, eta):
        self.eta = eta
        self.p = 0.9
        self.eps = 1e-6
        self.r = 0

    def update(self, theta, grad):
        grad2 = np.multiply(grad, grad)
        self.r = self.p * self.r + (1-self.p) * grad2
        alpha = self.eta / np.sqrt(self.eps + self.r)
        theta = theta - alpha * grad
        return theta

class Adam(GDOptimizer):
    def __init__(self, eta=0.001):
        self.eta = eta
        self.p1 = 0.9
        self.p2 = 0.999
        self.eps = 1e-8
        #self.s = np.zeros(shape)
        #self.r = np.zeros(shape)
        self.t = 0
        self.m = 0
        self.v = 0

    def update(self, theta, grad):
        self.t = self.t + 1
        self.m = self.p1 * self.m + (1-self.p1) * grad
        self.v = self.p2 * self.v + (1-self.p2) * np.multiply(grad, grad)
        m_hat = self.m / (1 - self.p1 ** self.t)
        v_hat = self.v / (1 - self.p2 ** self.t)
        d_theta = self.eta * m_hat / (self.eps + np.sqrt(v_hat))
        theta = theta - d_theta
        return theta

# 本算法要求在训练过程中两次介入：1. 在前向计算之前，先更新一次临时梯度 pre_update()；2. 在反向传播之后，再更新一次梯度 final_update()
class Nag(GDOptimizer):
    def __init__(self, eta):
        self.vt = 0
        self.eta = eta
        self.alpha = 0.9

    # 先用预测的梯度来更新W,b
    def pre_update(self, theta):
        theta_hat = theta - self.alpha * self.vt
        return theta_hat

    # 再用动量法更新W,b do final update
    def update(self, theta, grad):
        self.vt = self.alpha * self.vt + self.eta * grad
        theta = theta - self.vt
        return theta
