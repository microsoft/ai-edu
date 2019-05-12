# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from MiniFramework.Layer import *

class BnLayer(CLayer):
    def __init__(self, input_size):
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        self.eps = 1e-5
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, input, train=True):
        assert(input.ndim == 2 or input.ndim == 4)  # fc or cv
        self.x = input
        # 公式6
        mu = np.mean(self.x, axis=0)
        # 公式7
        self.x_mu  = self.x - mu
        self.var = np.mean(self.x_mu**2, axis=0) + self.eps
        # 公式8
        self.std = np.sqrt(self.var)
        self.norm_x = self.x_mu / self.std
        # 公式9
        self.z = self.gamma * self.norm_x + self.beta
        return self.z

    def backward(self, delta_in):
        assert(delta_in.ndim == 2 or delta_in.ndim == 4)  # fc or cv
        m = self.x.shape[0]
        # calculate d_beta, b_gamma
        self.d_beta = np.sum(delta_in, axis=0)
        self.d_gamma = np.sum(delta_in * self.norm_x, axis=0)
        # calculate delta_out
        # 公式14
        d_norm_x = self.gamma * delta_in 
        # 公式16
        d_var = -0.5 * np.sum(d_norm_x * self.x_mu, axis=0) / (self.var * self.std)
        # 公式18
        d_mu = -np.sum(d_norm_x / self.std, axis=0) - 2 / m * d_var * np.sum(self.x_mu, axis=0)
        # 公式13
        delta_out = d_norm_x / self.std + d_var * 2 * self.x_mu / m + d_mu / m
        return delta_out, self.d_gamma, self.d_beta
        
    def update(self, learning_rate=0.1):
        self.gamma = self.gamma - self.d_gamma * learning_rate
        self.beta = self.beta - self.d_beta * learning_rate

