# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.DataReader_2_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.LossFunction_1_1 import *
from MiniFramework.TrainingHistory_3_0 import *

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
        self.W = W
        self.x = x
        # 公式1
        self.h = np.dot(self.x, U) + bh
        # 公式2
        self.s = Tanh().forward(self.h)
        self.z = 0

    def backward(self, y, dh_t2):
        # 公式14
        self.dh = np.dot(dh_t2, self.W.T) * Tanh().backward(self.s)
         # 公式15
        self.dbh = self.dh       
        # 公式16
        self.dbz = 0
        # 公式17
        self.dU = np.dot(self.x.T, self.dh)
        # 公式18
        self.dV = 0        
        # 公式19
        self.dW = 0

class timestep_2(object):
    def forward(self,x,U,V,W,bh,bz,s_t1):
        self.V = V
        self.x = x
        # 公式3
        self.h = np.dot(x, U) + np.dot(s_t1, W) + bh
        # 公式4
        self.s = Tanh().forward(self.h)
        # 公式5
        self.z = np.dot(self.s, V) + bz

    def backward(self, y, s_t1):
        # 公式7
        self.dz = self.z - y
        # 公式8
        self.dh = np.dot(self.dz, self.V.T) * Tanh().backward(self.s)
        # 公式9
        self.dbz = self.dz
        # 公式10
        self.dbh = self.dh        
        # 公式11
        self.dV = np.dot(self.s.T, self.dz)
        # 公式12
        self.dU = np.dot(self.x.T, self.dh)
        # 公式13
        self.dW = np.dot(s_t1.T, self.dh)


class net(object):
    def __init__(self, dr):
        self.dr = dr
        self.loss_fun = LossFunction_1_1(NetType.Fitting)
        self.loss_trace = TrainingHistory_3_0()
        self.t1 = timestep_1()
        self.t2 = timestep_2()

    def check_loss(self):
        X,Y = self.dr.GetValidationSet()
        self.t1.forward(X[:,0],self.U,self.V,self.W,self.bh)
        self.t2.forward(X[:,1],self.U,self.V,self.W,self.bh,self.bz,self.t1.s)
        loss2,acc2 = self.loss_fun.CheckLoss(self.t2.z,Y[:,1:2])
        return loss2,acc2

    def train(self):
        num_input = 1
        num_hidden = 1
        num_output = 1
        max_epoch = 100
        eta = 0.1
        self.U = np.random.normal(size=(num_input,num_hidden))
        self.W = np.random.normal(size=(num_hidden,num_hidden))
        self.V = np.random.normal(size=(num_hidden,num_output))
        self.bh = np.zeros((1,num_hidden))
        self.bz = np.zeros((1,num_output))
        max_iteration = self.dr.num_train
        for epoch in range(max_epoch):
            for iteration in range(max_iteration):
                # get data
                batch_x, batch_y = self.dr.GetBatchTrainSamples(1, iteration)
                xt1 = batch_x[:,0,:]
                xt2 = batch_x[:,1,:]
                yt1 = batch_y[:,0]
                yt2 = batch_y[:,1]
                # forward
                self.t1.forward(xt1,self.U,self.V,self.W,self.bh)
                self.t2.forward(xt2,self.U,self.V,self.W,self.bh,self.bz,self.t1.s)
                # backward
                self.t2.backward(yt2, self.t1.s)
                self.t1.backward(yt1, self.t2.dh)
                # update
                self.U = self.U - (self.t1.dU + self.t2.dU)*eta
                self.V = self.V - (self.t1.dV + self.t2.dV)*eta
                self.W = self.W - (self.t1.dW + self.t2.dW)*eta
                self.bh = self.bh - (self.t1.dbh + self.t2.dbh)*eta
                self.bz = self.bz - (self.t1.dbz + self.t2.dbz)*eta
            #end for
            total_iteration = epoch * max_iteration + iteration
            if (epoch % 1 == 0):
                #loss_train,acc_train = self.loss_fun.CheckLoss(self.t2.z,batch_y[:,1:2])
                loss_vld,acc_vld = self.check_loss()
                self.loss_trace.Add(epoch, total_iteration, None, None, loss_vld, acc_vld, None)
                print(epoch)
                #print(str.format("train:      loss={0:6f}, acc={1:6f}", loss_train, acc_train))
                print(str.format("validation: loss={0:6f}, acc={1:6f}", loss_vld, acc_vld))
        #end for
        # print parameters
        print(str.format("U={0}, bh={1},", self.U, self.bh))
        print(str.format("V={0}, bz={1},", self.V, self.bz))
        print(str.format("W={0}", self.W))
        # show loss history
        self.loss_trace.ShowLossHistory("Loss and Accuracy", XCoordinate.Epoch)

    def test(self):
        print("testing...")
        X,Y = self.dr.GetTestSet()
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
    
