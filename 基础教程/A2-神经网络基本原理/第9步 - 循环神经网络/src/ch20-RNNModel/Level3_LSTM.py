# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
import math
from MiniFramework.LSTMCell_1_2 import *

class LSTM(object):
    def __init__(self, input_size, hidden_size, num_layers, time_steps, batch_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.bias = bias
        self.buildlstm()
        self.init_lstm_params()

    def buildlstm(self):
        self.cells = []
        # Add cells[0]and cells[self.time_steps+1] for each layer, used to store h0,c0 and dh(time_steps+1) for backward
        for i in range((self.time_steps+2) * self.num_layers):
            self.cells.append(LinearCell_1_2(self.input_size, self.hidden_size, self.bias))
        self.cells = np.asarray(self.cells).reshape(self.num_layers, (self.time_steps+2))
        for j in range(self.num_layers):
            self.cells[j][0].h = np.zeros((1, self.hidden_size))
            self.cells[j][0].c = np.zeros((1, self.hidden_size))
            self.cells[j][self.time_steps+1].dh = np.zeros((1, self.hidden_size))

    def init_lstm_params(self):
        self.W = []
        self.U = []
        self.b = []
        for i in range(self.num_layers):
            w = self.init_params((4 * self.hidden_size, self.hidden_size), "uniform")
            u = self.init_params((4 * self.input_size, self.hidden_size), "uniform")
            b = np.zeros((4, self.hidden_size))
            self.W.append(w)
            self.U.append(u)
            self.b.append(b)

    def init_params(self, shape, mode):
        p = []
        if mode == "uniform":
            std = 1.0 / math.sqrt(shape)
            p = np.random.uniform(-std, std, shape)
        elif mode == "random":
            p = np.random.random(shape)
        else:
            raise ValueError("Unsupported mode: " + mode)
        return p

    def forward(self, X):
        # expand X from [0, time_steps-1] to [0, time_steps+1], only use times from [1, time_steps]
        input = np.insert(X, 0, 0, axis=-1)
        input = np.insert(input, input.shape[-1], 0, axis=-1)

        for i in range(self.num_layers):
            for j in range(1, self.time_steps+1):
                self.cells[i][j].forward(input[:,j], self.cells[i][j-1].h, self.cells[i][j-1].c, self.W[i], self.U[i], self.b[i])
                input[:,j] = self.cells[i][j].h


    def backward(self, Y, dZ):
        # expand dZ from [0, time_steps-1] to [0, time_steps+1], only use times from [1, time_steps]
        dx = np.insert(dZ, 0, 0, axis=-1)
        dx = np.insert(dx, dx.shape[-1], 0, axis=-1)
        # backward
        for i in range(self.num_layers-1, -1, -1): # the index starts from 0 to num_layers-1
            for j in range(self.time_steps, 0, -1): # the index starts from 1 to time_steps
                in_grad = dx[:,j] + self.cells[i][j+1].dh
                self.cells[i][j].backward(self.cells[i][j-1].h, self.cells[i][j-1].c, in_grad)
                dx[:,j] = self.cells[i][j].dx

    def update(self, lr):
        for i in range(self.num_layers):
            for j in range(1, self.time_steps+1):
                self.U[i] = self.U[i] - self.cells[i][j].dU * lr / self.batch_size
                self.W[i] = self.W[i] - self.cells[i][j].dW * lr / self.batch_size
                if self.bias:
                    self.b[i] = self.b[i] - self.cells[i][j].db * lr / self.batch_size
