# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np
import math

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.DataReader_2_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.LossFunction_1_1 import *
from MiniFramework.TrainingHistory_3_0 import *
from MiniFramework.HyperParameters_4_3 import *

train_file = "../../data/ch19.train_minus.npz"
test_file = "../../data/ch19.test_minus.npz"

def load_data():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.Shuffle()
    dr.GenerateValidationSet(k=0)
    return dr

class timestep(object):
    # for the first cell, prev_s should be zero
    def forward(self,x,U,V,W,prev_s):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        # 公式6
        self.h = np.dot(x, U) + np.dot(prev_s, W)
        # 公式2
        self.s = Tanh().forward(self.h)
        # 公式3
        self.z = np.dot(self.s, V)
        # 公式4
        self.a = Logistic().forward(self.z)

    # for the first cell, prev_s should be zero
    # for the last cell, next_dh should be zero
    def backward(self, y, prev_s, next_dh):
        # 公式7
        self.dz = (self.a - y)
        # 公式11
        # if this is the last cell, next_dh should be zero
        self.dh = (np.dot(self.dz, self.V.T) + np.dot(next_dh, self.W.T)) * Tanh().backward(self.s)
        # 公式12
        self.dV = np.dot(self.s.T, self.dz)
        # 公式13
        self.dU = np.dot(self.x.T, self.dh)
        # 公式15
        # if this is the first cell, then prev_s is zero, dW will be zero
        self.dW = np.dot(prev_s.T, self.dh)


class net(object):
    def __init__(self, hp):
        self.hp = hp
        self.U = np.random.random((self.hp.num_input, self.hp.num_hidden))
        self.W = np.random.random((self.hp.num_hidden, self.hp.num_hidden))
        self.V = np.random.random((self.hp.num_hidden, self.hp.num_output))
        self.zero_state = np.zeros((self.hp.batch_size, self.hp.num_hidden))
        self.loss_fun = LossFunction_1_1(self.hp.net_type)
        self.loss_trace = TrainingHistory_3_0()
        self.ts_list = []
        for i in range(self.hp.num_step+1):
            ts = timestep()
            self.ts_list.append(ts)
        #end for
        self.ts_list[self.hp.num_step].s = np.zeros((self.hp.batch_size, self.hp.num_hidden))
        self.ts_list[self.hp.num_step].dh = np.zeros((self.hp.batch_size, self.hp.num_hidden))

    def forward(self,X):
        for i in range(0,self.hp.num_step):
            self.ts_list[i].forward(X[:,i],self.U,self.V,self.W,self.ts_list[i-1].s)
        #end for

    def backward(self,Y):
        for i in range(self.hp.num_step-1, -1, -1):
            self.ts_list[i].backward(Y[:,i], self.ts_list[i-1].s, self.ts_list[i+1].dh)
        #end for

    def update(self):
        du = np.zeros_like(self.U)
        dv = np.zeros_like(self.V)
        dw = np.zeros_like(self.W)
        for i in range(self.hp.num_step):
            du += self.ts_list[i].dU
            dv += self.ts_list[i].dV
            dw += self.ts_list[i].dW
        #end for
        self.U = self.U - du * self.hp.eta
        self.V = self.V - dv * self.hp.eta
        self.W = self.W - dw * self.hp.eta

    def check_loss(self,X,Y):
        self.forward(X)
        LOSS = 0
        output = None
        for i in range(self.hp.num_step):
            loss,_ = self.loss_fun.CheckLoss(self.ts_list[i].a,Y[:,i:i+1])
            LOSS += loss
            if (output is None):
                output = self.ts_list[i].a
            else:
                output = np.concatenate((output, self.ts_list[i].a), axis=1)
        result = np.round(output).astype(int)
        correct = 0
        for i in range(X.shape[0]):
            if (np.allclose(result[i], Y[i])):
                correct += 1
        acc = correct/X.shape[0]
        return LOSS/4,acc,result

    def train(self, dataReader, checkpoint=0.1):
        max_iteration = math.ceil(dataReader.num_train/self.hp.batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))

        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get data
                batch_x, batch_y = dataReader.GetBatchTrainSamples(1, iteration)
                # forward
                self.forward(batch_x)
                # backward
                self.backward(batch_y)
                # update
                self.update()
                # check loss
                total_iteration = epoch * max_iteration + iteration               
                if (total_iteration+1) % checkpoint_iteration == 0:
                    X,Y = dataReader.GetValidationSet()
                    loss,acc,_ = self.check_loss(X,Y)
                    self.loss_trace.Add(epoch, total_iteration, None, None, loss, acc, None)
                    print(epoch, total_iteration)
                    print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))
                #end if
            #enf for
            if (acc == 1.0):
                break
        #end for
        self.loss_trace.ShowLossHistory("Loss and Accuracy", XCoordinate.Iteration)

    def test(self, dataReader):
        print("testing...")
        X,Y = dataReader.GetTestSet()
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
        #end for

def reverse(a):
    l = a.tolist()
    l.reverse()
    return l

if __name__=='__main__':
    dataReader = load_data()
    eta = 0.1
    max_epoch = 100
    batch_size = 1
    num_step = 4
    num_input = 2
    num_output = 1
    num_hidden = 8
    hp = HyperParameters_4_3(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden, num_output, 
        NetType.Fitting)
    n = net(hp)
    n.train(dataReader, checkpoint=0.1)
    n.test(dataReader)
