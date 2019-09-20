# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.DataReader_2_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.LossFunction_1_1 import *

train_file = "../../data/ch19.train_minus.npz"
test_file = "../../data/ch19.test_minus.npz"

def load_data():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.Shuffle()
    dr.GenerateValidationSet(k=0)
    return dr

class timestep(object):
    def forward(self,x,h_prev,U,V,W,bz,ba):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        self.z = np.dot(x, U) + np.dot(h_prev, W) #+ bz
        self.h = Tanh().forward(self.z)
        self.a = np.dot(self.h, V) #+ ba
        self.output = Logistic().forward(self.a)

    def backward(self, y, h_prev, dz_next):
        self.da = (self.output - y)
        self.dz = (np.dot(self.da, self.V.T) + np.dot(dz_next, self.W.T)) * Tanh().backward(self.h)
        self.dV = np.dot(self.h.T, self.da)
        self.dU = np.dot(self.x.T, self.dz)
        self.dW = np.dot(h_prev.T, self.dz)
        self.dba = self.da
        self.dbz = self.dz

class timestep_1(timestep):
    # compare with timestep class: no h_t value from previous layer
    def forward(self,x,U,V,W,bz,ba):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        self.z = np.dot(self.x, U) #+ bz
        self.h = Tanh().forward(self.z)
        self.a = np.dot(self.h, V) #+ ba
        self.output = Logistic().forward(self.a)

    def backward(self, y, dz_next):
        self.da = (self.output - y)
        self.dz = (np.dot(self.da, self.V.T) + np.dot(dz_next, self.W.T)) * Tanh().backward(self.h)
        self.dV = np.dot(self.h.T, self.da)
        self.dU = np.dot(self.x.T, self.dz)
        self.dW = 0
        self.dba = self.da
        self.dbz = self.dz

class timestep_4(timestep):
    # compare with timestep class: no dz_t from future layer
    def backward(self, y, h_prev):
        self.da = self.output - y
        self.dz = np.dot(self.da, self.V.T) * Tanh().backward(self.h)
        self.dV = np.dot(self.h.T, self.da)
        self.dU = np.dot(self.x.T, self.dz)
        self.dW = np.dot(h_prev.T, self.dz)
        self.dba = self.da
        self.dbz = self.dz

class net(object):
    def __init__(self, dr):
        self.dr = dr
        self.loss_fun = LossFunction_1_1(NetType.BinaryClassifier)
        self.t1 = timestep_1()
        self.t2 = timestep()
        self.t3 = timestep()
        self.t4 = timestep_4()

    def forward(self,X):
        self.t1.forward(X[:,0],          self.U,self.V,self.W,self.bz,self.ba)
        self.t2.forward(X[:,1],self.t1.h,self.U,self.V,self.W,self.bz,self.ba)
        self.t3.forward(X[:,2],self.t2.h,self.U,self.V,self.W,self.bz,self.ba)
        self.t4.forward(X[:,3],self.t3.h,self.U,self.V,self.W,self.bz,self.ba)

    def backward(self,Y):
        self.t4.backward(Y[:,3], self.t3.h)
        self.t3.backward(Y[:,2], self.t2.h, self.t4.dz)
        self.t2.backward(Y[:,1], self.t1.h, self.t3.dz)
        self.t1.backward(Y[:,0],            self.t2.dz)

    def check_loss(self,X,Y):
        self.forward(X)
        loss1,acc1 = self.loss_fun.CheckLoss(self.t1.output,Y[:,0:1])
        loss2,acc2 = self.loss_fun.CheckLoss(self.t2.output,Y[:,1:2])
        loss3,acc3 = self.loss_fun.CheckLoss(self.t3.output,Y[:,2:3])
        loss4,acc4 = self.loss_fun.CheckLoss(self.t4.output,Y[:,3:4])
        output = np.concatenate((self.t1.output,self.t2.output,self.t3.output,self.t4.output), axis=1)
        result = np.round(output).astype(int)
        correct = 0
        for i in range(X.shape[0]):
            if (np.allclose(result[i], Y[i])):
                correct += 1
        acc = correct/X.shape[0]
        loss = (loss1 + loss2 + loss3 + loss4)/4
        return loss,acc,result

    def train(self):
        num_input = 2
        num_hidden = 3
        num_output = 1
        max_epoch = 100
        eta = 0.1
        self.U = np.random.random((num_input,num_hidden))*2-1
        self.W = np.random.random((num_hidden,num_hidden))*2-1
        self.V = np.random.random((num_hidden,num_output))*2-1
        self.bz = np.zeros((1,num_hidden))
        self.ba = np.zeros((1,num_output))
        for epoch in range(max_epoch):
            dr.Shuffle()
            for i in range(dr.num_train):
                # get data
                batch_x, batch_y = self.dr.GetBatchTrainSamples(1, i)
                # forward
                self.forward(batch_x)
                self.backward(batch_y)
                # update
                self.U = self.U - (self.t1.dU + self.t2.dU + self.t3.dU + self.t4.dU)*eta
                self.V = self.V - (self.t1.dV + self.t2.dV + self.t3.dV + self.t4.dV)*eta
                self.W = self.W - (self.t1.dW + self.t2.dW + self.t3.dW + self.t4.dW)*eta
                self.ba = self.ba - (self.t1.dba + self.t2.dba + self.t3.dba + self.t4.dba)*eta
                self.bz = self.bz - (self.t1.dbz + self.t2.dbz + self.t3.dbz + self.t4.dbz)*eta
            #end for
            if (epoch % 1 == 0):                
                X,Y = dr.GetValidationSet()
                loss,acc,_ = self.check_loss(X,Y)
                print(epoch)
                print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))
                if (acc == 1.0):
                    break
        #end for

    def test(self):
        print("testing...")
        X,Y = dr.GetTestSet()
        count = X.shape[0]
        loss,acc,result = self.check_loss(X,Y)
        print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))
        r = np.random.randint(0,count,10)
        for i in range(10):
            idx = r[i]
            x1 = X[idx,:,0]
            x2 = X[idx,:,1]
            print("  x1:", reverse(x1))
            print("- x2:", reverse(x2))
            print("------------------")
            print("true:", reverse(Y[idx]))
            print("pred:", reverse(result[idx]))
            print("====================")

def reverse(a):
    l = a.tolist()
    l.reverse()
    return l

if __name__=='__main__':
    dr = load_data()
    count = dr.num_train
    n = net(dr)
    n.train()
    n.test()
