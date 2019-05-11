# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from MiniFramework.Layer import *

class BnLayer(CLayer):
    def __init__(self):
        self.gamma = np.ones()
        self.beta = np.zeros()
        self.epsilon = 1e-5


    def forward(self, input, train=True):
        assert(input.ndim == 2 or input.ndim == 4)  # fc or cv
        self.x = input
        mu = np.mean(self.x, axis=0)
        self.x_mu  = self.x - mu
        self.var = np.mean(self.x_mu**2, axis=0)
        self.std = np.sqrt(self.var + self.epsilon)
        self.norm_x = self.x_mu / self.std
        self.z = self.gamma * self.norm_x + self.beta
        return self.z

    def backward(self, delta_in):
        assert(delta_in.ndim == 2 or delta_in.ndim == 4)  # fc or cv
        m = self.x.shape[0]
        # calculate d_beta, b_gamma
        d_beta = np.sum(delta_in, axis=0)
        d_gamma = np.sum(delta_in * self.norm_x, axis=0)
        # calculate delta_out
        d_norm_x = delta_in * self.gamma
        d_var = np.sum(d_norm_x * self.x_mu, axis=0) * (-0.5) / ((self.var + self.epsilon) * self.std)
        d_mu = -np.sum(d_norm_x / self.std) - 2 * d_var * np.sum(self.x_mu, axis=0) /m
        delta_out = d_norm_x / self.std + d_var * 2 * self.x_mu / m + d_mu / m
        return delta_out