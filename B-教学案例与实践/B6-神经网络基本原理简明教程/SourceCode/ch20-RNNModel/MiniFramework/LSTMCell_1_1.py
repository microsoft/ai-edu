# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
This version is 1.1
Implement the basic functions of lstm cell and linear cell
Can't support batch input
"""

import sys
import math
import numpy as np
from MiniFramework.Layer import *
from MiniFramework.ActivationLayer import *

class LSTMCell_1_1(object):
    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dW = np.zeros((4 * self.hidden_size, self.hidden_size))
        self.dU = np.zeros((4 * self.input_size, self.hidden_size))
        self.db = np.zeros((4, self.hidden_size))

    def get_params(self, W, U, b):
        self.wf = W[0: self.hidden_size,:]
        self.wi = W[self.hidden_size: 2 * self.hidden_size,:]
        self.wg = W[2 * self.hidden_size: 3 * self.hidden_size,:]
        self.wo = W[3 * self.hidden_size: 4 * self.hidden_size,:]

        self.uf = U[0: self.input_size,:]
        self.ui = U[self.input_size: 2 * self.input_size,:]
        self.ug = U[2 * self.input_size: 3 * self.input_size,:]
        self.uo = U[3 * self.input_size: 4 * self.input_size,:]

        if self.bias:
            self.bf = b[0,:]
            self.bi = b[1,:]
            self.bg = b[2,:]
            self.bo = b[3,:]
        else:
            self.bf = self.bi = self.bg = self.bo = np.zeros((1, self.hidden_size))

    def forward(self, x, h_p, c_p, W, U, b):
        self.get_params(W, U, b)
        self.x = x

        self.f = self.get_gate(x, h_p, self.wf, self.uf, self.bf, Sigmoid())
        self.i = self.get_gate(x, h_p, self.wi, self.ui, self.bi, Sigmoid())
        self.g = self.get_gate(x, h_p, self.wg, self.ug, self.bg, Tanh())
        self.o = self.get_gate(x, h_p, self.wo, self.uo, self.bo, Sigmoid())

        self.c = np.multiply(self.f, c_p) + np.multiply(self.i, self.g)
        self.h = np.multiply(self.o, Tanh().forward(self.c))

    def get_gate(self, x, h, W, U, b, activator, bias=True):
        if self.bias:
            z = np.dot(h, W) + np.dot(x, U) + b
        else:
            z = np.dot(h, W) + np.dot(x, U)
        a = activator.forward(z)
        return a

    def backward(self, h_p, c_p, in_grad):
        tanh = lambda x : Tanh().forward(x)
        self.dzo = in_grad * tanh(self.c) * self.o * (1 - self.o)
        self.dc = in_grad * self.o * (1 - tanh(self.c) * tanh(self.c))
        self.dzg = self.dc * self.i * (1- self.g * self.g)
        self.dzi = self.dc * self.g * self.i * (1 - self.i)
        self.dzf = self.dc * c_p * self.f * (1 - self.f)

        self.dW[3 * self.hidden_size: 4 * self.hidden_size] = np.dot(h_p.T, self.dzo)
        self.dW[2 * self.hidden_size: 3 * self.hidden_size] = np.dot(h_p.T, self.dzg)
        self.dW[self.hidden_size: 2 * self.hidden_size] = np.dot(h_p.T, self.dzi)
        self.dW[0: self.hidden_size] = np.dot(h_p.T, self.dzf)

        self.dU[3 * self.input_size: 4 * self.input_size] = np.dot(self.x.T, self.dzo)
        self.dU[2 * self.input_size: 3 * self.input_size] = np.dot(self.x.T, self.dzg)
        self.dU[self.input_size: 2 * self.input_size] = np.dot(self.x.T, self.dzi)
        self.dU[0: self.input_size] = np.dot(self.x.T, self.dzf)

        if self.bias:
            self.db[0] = self.dzo
            self.db[1] = self.dzg
            self.db[2] = self.dzi
            self.db[3] = self.dzf

        # pass to previous time step
        self.dh = np.dot(self.dzf, self.wf.T) + np.dot(self.dzi, self.wi.T) + np.dot(self.dzg, self.wg.T) + np.dot(self.dzo, self.wo.T)
        # pass to previous layer
        self.dx = np.dot(self.dzf, self.uf.T) + np.dot(self.dzi, self.ui.T) + np.dot(self.dzg, self.ug.T) + np.dot(self.dzo, self.uo.T)


class LinearCell_1_1(object):
    def __init__(self, input_size, output_size, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias

    def forward(self, x, V, b):
        self.x = x
        self.V = V
        if self.bias:
            self.b = b
        else:
            self.b = np.zeros((1,self.output_size))
        self.z = np.dot(x, V) + b

    def backward(self, in_grad):
        self.dz = in_grad.reshape(1,-1)
        self.dV = np.dot(self.x.T, self.dz)
        if self.bias:
            self.db = self.dz
        self.dx = np.dot(self.dz, self.V.T)