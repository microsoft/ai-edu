
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
from MiniFramework.LSTMCell_1_2 import *
from MiniFramework.GRUCell import *

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
        self.grucell = []
        self.linearcell = []
        for i in range(self.times):
            self.grucell.append(GRUCell(input_size, hidden_size))
            self.linearcell.append(LinearCell_1_2(hidden_size, output_size, Logistic(), bias=False))

    def forward(self, X):
        hp = np.zeros((1, self.hidden_size))
        for i in range(self.times):
            self.grucell[i].forward(X[:,i], hp, self.W, self.U)
            hp = self.grucell[i].h
            self.linearcell[i].forward(hp, self.V)

    def backward(self, Y):
        hp = []
        # The last time step:
        tl = self.times-1
        dz = self.linearcell[tl].a - Y[:,tl:tl+1]
        self.linearcell[tl].backward(dz)
        hp = self.grucell[tl-1].h
        self.grucell[tl].backward(hp, self.linearcell[tl].dx)
        # Middle time steps:
        dh = []
        for i in range(tl-1, 0, -1):
            dz = self.linearcell[i].a - Y[:,i:i+1]
            self.linearcell[i].backward(dz)
            hp = self.grucell[i-1].h
            dh = self.linearcell[i].dx + self.grucell[i+1].dh
            self.grucell[i].backward(hp, dh)
        # The first time step:
        dz = self.linearcell[0].a - Y[:,0:1]
        self.linearcell[0].backward(dz)
        dh = self.linearcell[0].dx + self.grucell[1].dh
        self.grucell[0].backward(np.zeros((self.batch_size, self.hidden_size)), dh)

    def check_loss(self, X, Y):
        self.forward(X)
        loss_list = np.zeros((self.times, self.output_size))
        acc_list = np.zeros((self.times, self.output_size))
        for i in range(self.times):
            loss_list[i], acc_list[i] = self.loss_fun.CheckLoss(self.linearcell[i].a, Y[:,i:(i+1)])
        output = np.concatenate((self.linearcell[0].a,self.linearcell[1].a,self.linearcell[2].a,self.linearcell[3].a), axis=1)
        result = np.round(output).astype(int)
        correct = 0
        for i in range(X.shape[0]):
            if (np.allclose(result[i], Y[i])):
                correct += 1
        acc = correct/X.shape[0]
        loss = np.mean(loss_list, axis=0)[0]
        return loss,acc,result

    def init_params_uniform(self, shape):
        p = []
        std = 1.0 / math.sqrt(self.hidden_size)
        p = np.random.uniform(-std, std, shape)
        return p


    def train(self, batch_size, checkpoint=0.1):
        self.batch_size = batch_size
        max_epoch = 100
        eta = 0.1
        # Try different initialize method
        # self.U = np.random.random((3 * self.input_size, self.hidden_size))
        # self.W = np.random.random((3 * self.hidden_size, self.hidden_size))
        self.U = self.init_params_uniform((3 * self.input_size, self.hidden_size))
        self.W = self.init_params_uniform((3 * self.hidden_size, self.hidden_size))
        self.V = np.random.random((self.hidden_size, self.output_size))


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
                    # self.grucell[i].merge_params()
                    self.U = self.U - self.grucell[i].dU * eta /self.batch_size
                    self.W = self.W - self.grucell[i].dW * eta /self.batch_size
                    self.V = self.V - self.linearcell[i].dV * eta /self.batch_size
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
            x1_dec = int("".join(map(str, reverse(x1))), 2)
            x2_dec = int("".join(map(str, reverse(x2))), 2)
            print("{0} - {1} = {2}".format(x1_dec, x2_dec, (x1_dec-x2_dec)))
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