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
from MiniFramework.LSTMCell import *

train_file = "../../data/ch19.train_minus.npz"
test_file = "../../data/ch19.test_minus.npz"

def load_data():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.Shuffle()
    dr.GenerateValidationSet(k=0)
    return dr

class net(object):
    def __init__(self, dr, input_size, hidden_size, output_size):
        self.dr = dr
        self.loss_fun = LossFunction_1_1(NetType.BinaryClassifier)
        self.loss_trace = TrainingHistory_3_0()
        self.times = 4
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstmcell = []
        self.linearcell = []
        self.a = []
        for i in range(self.times):
            self.lstmcell.append(LSTMCell(input_size, hidden_size))
            self.linearcell.append(LinearCell(hidden_size, output_size))
            self.a.append((1, self.output_size))

    def forward(self, X):
        hp = np.zeros((1, self.hidden_size))
        cp = np.zeros((1, self.hidden_size))
        for i in range(self.times):
            self.lstmcell[i].forward(X[:,i], hp, cp, self.W, self.U, self.bh)
            hp = self.lstmcell[i].h
            cp = self.lstmcell[i].c
            self.linearcell[i].forward(hp, self.V, self.b)
            self.a[i] = Logistic().forward(self.linearcell[i].z)

    def backward(self, Y):
        dh = np.zeros((1,self.hidden_size))
        for i in range((self.times -1) , -1, -1):
            if i == 0:
                hp = np.zeros((1, self.hidden_size))
                cp = np.zeros((1, self.hidden_size))
            else:
                hp = self.lstmcell[i-1].h
                cp = self.lstmcell[i-1].c
            dz = self.a[i] - Y[:,i]
            self.linearcell[i].backward(dz)
            dh = dh + self.linearcell[i].dx
            self.lstmcell[i].backward(hp, cp, dh)
            dh = self.lstmcell[i].dh

    def check_loss(self, X, Y):
        self.forward(X)
        loss_list = np.zeros((self.times, self.output_size))
        acc_list = np.zeros((self.times, self.output_size))
        for i in range(self.times):
            loss_list[i], acc_list[i] = self.loss_fun.CheckLoss(self.a[i], Y[:,i:(i+1)])
        output = np.concatenate((self.a[0],self.a[1],self.a[2],self.a[3]), axis=1)
        result = np.round(output).astype(int)
        correct = 0
        for i in range(X.shape[0]):
            if (np.allclose(result[i], Y[i])):
                correct += 1
        acc = correct/X.shape[0]
        loss = np.mean(loss_list, axis=0)[0]
        return loss,acc,result

    def train(self, batch_size, checkpoint=0.1):
        max_epoch = 40
        eta = 0.1
        self.U = np.random.random((4 * self.input_size, self.hidden_size))
        self.W = np.random.random((4 * self.hidden_size, self.hidden_size))
        self.V = np.random.random((self.hidden_size, self.output_size))
        self.bh = np.zeros((4, self.hidden_size))
        self.b = np.zeros((1, self.output_size))

        max_iteration = math.ceil(self.dr.num_train/batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))

        for epoch in range(max_epoch):
            self.dr.Shuffle()
            for iteration in range(max_iteration):
                # get data
                batch_x, batch_y = self.dr.GetBatchTrainSamples(batch_size, iteration)
                # forward
                self.forward(batch_x)
                self.backward(batch_y)
                # update
                for i in range(self.times):
                    self.U = self.U - self.lstmcell[i].dU * eta
                    self.W = self.W - self.lstmcell[i].dW * eta
                    self.bh = self.bh - self.lstmcell[i].db * eta
                    self.V = self.V - self.linearcell[i].dV * eta
                    self.b = self.b - self.linearcell[i].db * eta
                # check loss
                total_iteration = epoch * max_iteration + iteration
                if (total_iteration+1) % checkpoint_iteration == 0:
                    X,Y = self.dr.GetValidationSet()
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
    input_size = 2
    hidden_size = 4
    output_size = 1
    n = net(dr, input_size, hidden_size, output_size)
    n.train(batch_size=1, checkpoint=0.1)
    n.test()