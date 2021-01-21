# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from MiniFramework.Layer import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *

class GRUCell(object):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

    def split_params(self, w, size):
        s=[]
        for i in range(3):
            s.append(w[(i*size):((i+1)*size)])
        return s[0], s[1], s[2]

    # Get shared parameters, and split them to fit 3 gates, in the order of z, r, \tilde{h} (n stands for \tilde{h} in code)
    def get_params(self, W, U):
            self.wz, self.wr, self.wn = self.split_params(W, self.hidden_size)
            self.uz, self.ur, self.un = self.split_params(U, self.input_size)

    def forward(self, x, h_p, W, U):
        self.get_params(W, U)
        self.x = x

        self.z = Sigmoid().forward(np.dot(h_p, self.wz) + np.dot(x, self.uz))
        self.r = Sigmoid().forward(np.dot(h_p, self.wr) + np.dot(x, self.ur))
        self.n = Tanh().forward(np.dot((self.r * h_p), self.wn) + np.dot(x, self.un))
        self.h = (1 - self.z) * h_p + self.z * self.n

    def backward(self, h_p, in_grad):
        self.dzz = in_grad * (self.n - h_p) * self.z * (1 - self.z)
        self.dzn = in_grad * self.z * (1 - self.n * self.n)
        self.dzr = np.dot(self.dzn, self.wn.T) * h_p * self.r * (1 - self.r)

        self.dwn = np.dot((self.r * h_p).T, self.dzn)
        self.dun = np.dot(self.x.T, self.dzn)
        self.dwr = np.dot(h_p.T, self.dzr)
        self.dur = np.dot(self.x.T, self.dzr)
        self.dwz = np.dot(h_p.T, self.dzz)
        self.duz = np.dot(self.x.T, self.dzz)

        self.merge_params()

        # pass to previous time step
        self.dh = in_grad * (1 - self.z) + np.dot(self.dzn, self.wn.T) * self.r + np.dot(self.dzr, self.wr.T) + np.dot(self.dzz, self.wz.T)
        # pass to previous layer
        self.dx = np.dot(self.dzn, self.un.T) + np.dot(self.dzr, self.ur.T) + np.dot(self.dzz, self.uz.T)

    def merge_params(self):
        self.dW = np.concatenate((self.dwz, self.dwr, self.dwn), axis=0)
        self.dU = np.concatenate((self.duz, self.dur, self.dun), axis=0)