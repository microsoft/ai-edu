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
    def __init__(self, dr, num_step, net_type):
        self.dr = dr
        self.num_step = num_step
        num_input = 2
        num_hidden = 4
        num_output = 1
        max_epoch = 100
        eta = 0.1
        self.U = np.random.random((num_input,num_hidden))
        self.W = np.random.random((num_hidden,num_hidden))
        self.V = np.random.random((num_hidden,num_output))

        self.loss_fun = LossFunction_1_1(net_type)
        self.loss_trace = TrainingHistory_3_0()
        self.ts_list = []
        for i in range(self.num_step):
            ts = timestep()
            ts_list.append(ts)

    def forward(self,X):
        for i in range(self.num_step):
            if (i == 0):
                self.ts_list[i].forward(X[:,i],self.U,self.V,self.W, None)
            else:
                self.ts_list[i].forward(X[:,i],self.U,self.V,self.W,self.ts_list[i-1].s)
            #end if
        #end for

    def backward(self,Y):
        for i in range(self.num_step-1, -1, -1):
            if (i == self.num_step-1):
                self.ts_list[i].backward(Y[:,i], self.ts_list[i-1].s, None)
            elif (i == 0):
                self.ts_list[i].backward(Y[:,i], None, self.ts_list[i+1].dh)
            else:
                self.ts_list[i].backward(Y[:,i], self.ts_list[i-1].s, self.ts_list[i+1].dh)
            #end if
        #end for

    def check_loss(self,X,Y):
        self.forward(X)
        LOSS = 0
        output = None
        for i in range(self.num_step):
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
        loss = (loss1 + loss2 + loss3 + loss4)/4
        return loss,acc,result

    def train(self, batch_size, checkpoint=0.1):
        
        max_iteration = math.ceil(self.dr.num_train/batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))

        for epoch in range(max_epoch):
            dr.Shuffle()
            for iteration in range(max_iteration):
                # get data
                batch_x, batch_y = self.dr.GetBatchTrainSamples(1, iteration)
                # forward
                self.forward(batch_x)
                self.backward(batch_y)
                # update
                self.U = self.U - (self.t1.dU + self.t2.dU + self.t3.dU + self.t4.dU)*eta
                self.V = self.V - (self.t1.dV + self.t2.dV + self.t3.dV + self.t4.dV)*eta
                self.W = self.W - (self.t1.dW + self.t2.dW + self.t3.dW + self.t4.dW)*eta
                # check loss
                total_iteration = epoch * max_iteration + iteration               
                if (total_iteration+1) % checkpoint_iteration == 0:
                    X,Y = dr.GetValidationSet()
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
        #end for

def reverse(a):
    l = a.tolist()
    l.reverse()
    return l

if __name__=='__main__':
    dr = load_data()
    count = dr.num_train
    n = net(dr)
    n.train(batch_size=1, checkpoint=0.1)
    n.test()
