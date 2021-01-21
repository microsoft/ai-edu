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

    def backward(self, y, prev_s, next_dh):
        # 公式7
        self.dz = (self.a - y)
        # 公式11
        self.dh = (np.dot(self.dz, self.V.T) + np.dot(next_dh, self.W.T)) * Tanh().backward(self.s)
        # 公式12
        self.dV = np.dot(self.s.T, self.dz)
        # 公式13
        self.dU = np.dot(self.x.T, self.dh)
        # 公式15
        self.dW = np.dot(prev_s.T, self.dh)

class timestep_1(timestep):
    # compare with timestep class: no h_t value from previous layer
    def forward(self,x,U,V,W):
        self.U = U
        self.V = V
        self.W = W
        self.x = x
        # 公式1
        self.h = np.dot(self.x, U)
        # 公式2
        self.s = Tanh().forward(self.h)
        # 公式3
        self.z = np.dot(self.s, V)
        # 公式4
        self.a = Logistic().forward(self.z)

    # for the first timestep, there has no prev_s
    def backward(self, y, next_dh):
        # 公式7
        self.dz = (self.a - y)
        # 公式11
        self.dh = (np.dot(self.dz, self.V.T) + np.dot(next_dh, self.W.T)) * Tanh().backward(self.s)
        # 公式12
        self.dV = np.dot(self.s.T, self.dz)
        # 公式13
        self.dU = np.dot(self.x.T, self.dh)
        # 公式14
        self.dW = 0

class timestep_4(timestep):
    # compare with timestep class: no next_dh from future layer
    def backward(self, y, prev_s):
        # 公式7
        self.dz = self.a - y
        # 公式9
        self.dh = np.dot(self.dz, self.V.T) * Tanh().backward(self.s)
        # 公式12
        self.dV = np.dot(self.s.T, self.dz)
        # 公式13
        self.dU = np.dot(self.x.T, self.dh)
        # 公式15
        self.dW = np.dot(prev_s.T, self.dh)

class net(object):
    def __init__(self, dr):
        self.dr = dr
        self.loss_fun = LossFunction_1_1(NetType.BinaryClassifier)
        self.loss_trace = TrainingHistory_3_0()
        self.t1 = timestep_1()
        self.t2 = timestep()
        self.t3 = timestep()
        self.t4 = timestep_4()

    def forward(self,X):
        self.t1.forward(X[:,0],self.U,self.V,self.W)
        self.t2.forward(X[:,1],self.U,self.V,self.W,self.t1.s)
        self.t3.forward(X[:,2],self.U,self.V,self.W,self.t2.s)
        self.t4.forward(X[:,3],self.U,self.V,self.W,self.t3.s)

    def backward(self,Y):
        self.t4.backward(Y[:,3], self.t3.s)
        self.t3.backward(Y[:,2], self.t2.s, self.t4.dh)
        self.t2.backward(Y[:,1], self.t1.s, self.t3.dh)
        self.t1.backward(Y[:,0],            self.t2.dh)

    def update(self, eta):
        self.U = self.U - (self.t1.dU + self.t2.dU + self.t3.dU + self.t4.dU)*eta
        self.V = self.V - (self.t1.dV + self.t2.dV + self.t3.dV + self.t4.dV)*eta
        self.W = self.W - (self.t1.dW + self.t2.dW + self.t3.dW + self.t4.dW)*eta

    def check_loss(self,X,Y):
        self.forward(X)
        loss1,acc1 = self.loss_fun.CheckLoss(self.t1.a,Y[:,0:1])
        loss2,acc2 = self.loss_fun.CheckLoss(self.t2.a,Y[:,1:2])
        loss3,acc3 = self.loss_fun.CheckLoss(self.t3.a,Y[:,2:3])
        loss4,acc4 = self.loss_fun.CheckLoss(self.t4.a,Y[:,3:4])
        # 只有四个时间步的输出全都正确时，才算作该样本预测正确
        # 假设4个时间步输入有3个正确，不能算作75%正确，而是0%
        output = np.concatenate((self.t1.a,self.t2.a,self.t3.a,self.t4.a), axis=1)
        result = np.round(output).astype(int)
        correct = 0
        for i in range(X.shape[0]):
            if (np.allclose(result[i], Y[i])):
                correct += 1
        acc = correct/X.shape[0]
        Loss = (loss1 + loss2 + loss3 + loss4)/4
        return Loss,acc,result

    def train(self, batch_size, checkpoint=0.1):
        num_input = 2
        num_hidden = 4
        num_output = 1
        max_epoch = 100
        eta = 0.1
        self.U = np.random.normal(size=(num_input, num_hidden))
        self.W = np.random.normal(size=(num_hidden,num_hidden))
        self.V = np.random.normal(size=(num_hidden,num_output))
        max_iteration = math.ceil(self.dr.num_train/batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))
        for epoch in range(max_epoch):
            dr.Shuffle()
            for iteration in range(max_iteration):
                # get data
                batch_x, batch_y = self.dr.GetBatchTrainSamples(1, iteration)
                # forward
                self.forward(batch_x)
                # backward
                self.backward(batch_y)
                # update
                self.update(eta)
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
