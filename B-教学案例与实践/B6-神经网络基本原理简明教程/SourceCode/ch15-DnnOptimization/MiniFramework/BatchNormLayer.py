# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

from MiniFramework.Layer import *

class BnLayer(CLayer):
    def __init__(self, input_size, momentum=0.9):
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        self.eps = 1e-5
        self.input_size = input_size
        self.output_size = input_size
        self.momentum = momentum
        self.running_mean = np.zeros((1,input_size))
        self.running_var = np.zeros((1,input_size))

    def forward(self, input, train=True):
        assert(input.ndim == 2 or input.ndim == 4)  # fc or cv
        self.x = input

        if train:
            # 公式6
            self.mu = np.mean(self.x, axis=0, keepdims=True)
            # 公式7
            self.x_mu  = self.x - self.mu
            self.var = np.mean(self.x_mu**2, axis=0, keepdims=True) + self.eps
            # 公式8
            self.std = np.sqrt(self.var)
            self.norm_x = self.x_mu / self.std
            # 公式9
            self.z = self.gamma * self.norm_x + self.beta
            # mean and var history, for test/inference
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.var
        else:
            self.mu = self.running_mean
            self.var = self.running_var
            self.norm_x = (self.x - self.mu) / np.sqrt(self.var + self.eps)
            self.z = self.gamma * self.norm_x + self.beta
        # end if
        return self.z

    def backward(self, delta_in, flag):
        assert(delta_in.ndim == 2 or delta_in.ndim == 4)  # fc or cv
        m = self.x.shape[0]
        # calculate d_beta, b_gamma
        # 公式11
        self.d_gamma = np.sum(delta_in * self.norm_x, axis=0, keepdims=True)
        # 公式12
        self.d_beta = np.sum(delta_in, axis=0, keepdims=True)

        # calculate delta_out
        # 公式14
        d_norm_x = self.gamma * delta_in 
        # 公式16
        d_var = -0.5 * np.sum(d_norm_x * self.x_mu, axis=0, keepdims=True) / (self.var * self.std) # == self.var ** (-1.5)
        # 公式18
        d_mu = -np.sum(d_norm_x / self.std, axis=0, keepdims=True) - 2 / m * d_var * np.sum(self.x_mu, axis=0, keepdims=True)
        # 公式13
        delta_out = d_norm_x / self.std + d_var * 2 * self.x_mu / m + d_mu / m
        if flag == -1:
            return delta_out, self.d_gamma, self.d_beta
        else:
            return delta_out
        
    def update(self, learning_rate=0.1):
        self.gamma = self.gamma - self.d_gamma * learning_rate
        self.beta = self.beta - self.d_beta * learning_rate

    def save_parameters(self, folder, name):
        file_name = str.format("{0}\\{1}.npz", folder, name)
        np.savez(file_name, gamma=self.gamma, beta=self.beta, mean=self.running_mean, var=self.running_var)

    def load_parameters(self, folder, name):
        file_name = str.format("{0}\\{1}.npz", folder, name)
        data = np.load(file_name)
        self.gamma = data["gamma"]
        self.beta = data["beta"]
        self.running_mean = data["running_mean"]
        self.running_var = data["running_var"]
