# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np

from MiniFramework.DataReader_2_0 import *
from MiniFramework.ActivationLayer import *

train_file = "../../data/ch19.train_echo.npz"
test_file = "../../data/ch19.test_echo.npz"

def load_data():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    return dr

def show(x, y, x_label, y_label):
    plt.plot(x, 'r-x', label=x_label)
    plt.plot(y, 'b-o', label=y_label)
    plt.legend()
    plt.show()

class timestep_1(object):
    def forward(self,x,U,V,W,bz,ba):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        self.z = np.dot(self.x, U) + bz
        self.h = Tanh().forward(self.z)
        self.a = np.dot(self.h, V) + ba

    def backward(self, y, dz_t2):
        self.da = self.a - y
        self.dz = (np.dot(self.da, self.V.T) + np.dot(dz_t2, self.W.T)) * Tanh().backward(self.h)
        self.dV = np.dot(self.h.T, self.da)
        self.dU = np.dot(self.x.T, self.dz)
        self.dW = np.dot(self.h.T, dz_t2)
        self.dba = self.da
        self.dbz = self.dz

class timestep_2(object):
    def forward(self,x,h_t1,U,V,W,bz,ba):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        self.z = np.dot(x, U) + np.dot(h_t1, W) + bz
        self.h = Tanh().forward(self.z)
        self.a = np.dot(self.h, V) + ba

    def backward(self, y):
        self.da = self.a - y
        self.dz = np.dot(self.da, self.V.T) * Tanh().backward(self.h)
        self.dV = np.dot(self.h.T, self.da)
        self.dU = np.dot(self.x.T, self.dz)
        self.dW = 0
        self.dba = self.da
        self.dbz = self.dz

def train(dr):
    num_hidden = 1
    max_epoch = 1000
    eta = 0.1
    U = np.random.random((1,num_hidden))
    V = np.random.random((num_hidden,1))
    W = np.random.random((num_hidden,num_hidden))
    bz = np.zeros((1,num_hidden))
    ba = np.zeros((1,1))
    count=200
    t1 = timestep_1()
    t2 = timestep_2()
    for epoch in range(max_epoch):
        for i in range(count-1):
            # get data
            batch_x, batch_y = dr.GetBatchTrainSamples(2, i)
            x1 = batch_x[0]
            x2 = batch_x[1]
            y1 = batch_y[0]
            y2 = batch_y[1]
            # forward
            t1.forward(x1,U,V,W,bz,ba)
            t2.forward(x2,t1.h,U,V,W,bz,ba)
            # backward
            t2.backward(y2)
            t1.backward(y1,t2.dz)
            # update
            U = U - (t1.dU + t2.dU)*eta
            V = V - (t1.dV + t2.dV)*eta
            W = W - (t1.dW + t2.dW)*eta
            ba = ba - (t1.dba + t2.dba)*eta
            bz = bz - (t1.dbz + t2.dbz)*eta
        #end for
        if (epoch % 10 == 0):
            loss = check_loss(t1,t2,
                          dr.XTrain[0:count-1].reshape(count-1,1),
                          dr.XTrain[1:count].reshape(count-1,1),
                          dr.YTrain[0:count-1].reshape(count-1,1),
                          dr.YTrain[1:count].reshape(count-1,1))
            print(loss)
    #end for



if __name__=='__main__':
    dr = load_data()
    show(dr.XTrain, dr.YTrain, "X", "Y")

    train(dr)
