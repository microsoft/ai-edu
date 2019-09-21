# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.DataReader_2_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.LossFunction_1_1 import *

train_file = "../../data/ch19.train_echo.npz"
test_file = "../../data/ch19.test_echo.npz"

def load_data():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.GenerateValidationSet(k=10)
    return dr

def show(x1, y1, x2, y2, x_label, y_label):
    plt.plot(x1, y1, 'r-x', label=x_label)
    plt.plot(x2, y2, 'b-o', label=y_label)
    plt.legend()
    plt.show()

class timestep_1(object):
    def forward(self,x,U,V,W,bh):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        self.h = np.dot(self.x, U) + bh
        self.s = Tanh().forward(self.h)
        self.z = 0

    def backward(self, y, dh_t2):
        #self.dz = self.z - y
        #self.dh = (np.dot(self.dz, self.V.T) + np.dot(dh_t2, self.W.T)) * Tanh().backward(self.s)
        self.dh = np.dot(dh_t2, self.W.T) * Tanh().backward(self.s)
        #self.dV = np.dot(self.s.T, self.dz)
        self.dV = 0
        self.dU = np.dot(self.x.T, self.dh)
        self.dW = 0
        self.dbz = 0
        self.dbh = self.dh

class timestep_2(object):
    def forward(self,x,U,V,W,bh,bz,s_t1):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        self.h = np.dot(x, U) + np.dot(s_t1, W) + bh
        self.s = Tanh().forward(self.h)
        self.z = np.dot(self.s, V) + bz

    def backward(self, y, s_t1):
        self.dz = self.z - y
        self.dh = np.dot(self.dz, self.V.T) * Tanh().backward(self.s)
        self.dV = np.dot(self.s.T, self.dz)
        self.dU = np.dot(self.x.T, self.dh)
        self.dW = np.dot(s_t1.T, self.dh)
        self.dbz = self.dz
        self.dbh = self.dh

class net(object):
    def __init__(self, dr):
        self.dr = dr
        self.loss_fun = LossFunction_1_1(NetType.Fitting)
        self.t1 = timestep_1()
        self.t2 = timestep_2()

    def check_loss(self,dr):
        X,Y = dr.GetValidationSet()
        self.t1.forward(X[:,0],self.U,self.V,self.W,self.bh)
        self.t2.forward(X[:,1],self.U,self.V,self.W,self.bh,self.bz,self.t1.s)
        #loss1,acc1 = loss_fun.CheckLoss(t1.a,y1)
        loss2,acc2 = self.loss_fun.CheckLoss(self.t2.z,Y[:,1:2])
        #loss = loss1 + loss2
        return loss2,acc2

    def train(self):
        num_input = 1
        num_hidden = 1
        num_output = 1
        max_epoch = 100
        eta = 0.1
        self.U = np.random.random((num_input,num_hidden))*2-1
        self.W = np.random.random((num_hidden,num_hidden))*2-1
        self.V = np.random.random((num_hidden,num_output))*2-1
        self.bh = np.zeros((1,num_hidden))
        self.bz = np.zeros((1,num_output))
        for epoch in range(max_epoch):
            for i in range(dr.num_train):
                # get data
                batch_x, batch_y = self.dr.GetBatchTrainSamples(1, i)
                xt1 = batch_x[:,0,:]
                xt2 = batch_x[:,1,:]
                yt1 = batch_y[:,0]
                yt2 = batch_y[:,1]
                # forward
                self.t1.forward(xt1,self.U,self.V,self.W,self.bh)
                self.t2.forward(xt2,self.U,self.V,self.W,self.bh,self.bz,self.t1.s)
                # backward
                self.t2.backward(yt2, self.t1.h)
                self.t1.backward(yt1, self.t2.dh)
                # update
                self.U = self.U - (self.t1.dU + self.t2.dU)*eta
                self.V = self.V - (self.t1.dV + self.t2.dV)*eta
                self.W = self.W - (self.t1.dW + self.t2.dW)*eta
                self.bh = self.bh - (self.t1.dbh + self.t2.dbh)*eta
                self.bz = self.bz - (self.t1.dbz + self.t2.dbz)*eta
            #end for
            if (epoch % 1 == 0):
                loss,acc = self.check_loss(dr)
                print(epoch)
                print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))
        #end for

    def test(self):
        print("testing...")
        X,Y = dr.GetTestSet()
        count = X.shape[0]
        self.t1.forward(X[:,0],self.U,self.V,self.W,self.bh)
        self.t2.forward(X[:,1],self.U,self.V,self.W,self.bh,self.bz,self.t1.s)
        loss2,acc2 = self.loss_fun.CheckLoss(self.t2.z,Y[:,1:2])
        print(str.format("loss={0:6f}, acc={1:6f}", loss2, acc2))
        show(range(0,count), X[:,0], range(0,count), self.t2.z,"test","predication")
        
if __name__=='__main__':
    dr = load_data()
    count = dr.num_train
    show(range(0,count), dr.XTrain[:,0,0], range(1,count+1), dr.YTrain[:,1], "X", "Y")
    n = net(dr)
    n.train()
    n.test()
