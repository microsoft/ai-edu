# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.LossFunction_1_1 import *
from MiniFramework.TrainingHistory_3_0 import *
from MiniFramework.HyperParameters_4_3 import *
from MiniFramework.WeightsBias_2_1 import *
from ExtendedDataReader.PM25DataReader import *


def load_data(net_type, num_step):
    dr = PM25DataReader(net_type, num_step)
    dr.ReadData()
    dr.Normalize()
    dr.GenerateValidationSet(k=1000)
    return dr

class timestep(object):
    # for the first cell, prev_s should be zero
    def forward(self, x, U, V, Q, W1, W2, prev_s1, prev_s2, isFirst, isLast):
        self.U = U
        self.V = V
        self.Q = Q
        self.W1 = W1
        self.W2 = W2
        self.x = x

        # level 1
        if (isFirst):
            # 公式1
            self.h1 = np.dot(x, U)
        else:
            # 公式3
            self.h1 = np.dot(x, U) + np.dot(prev_s1, W1) 
        # 公式5
        self.s1 = Tanh().forward(self.h1)

        # level 2
        if (isFirst):
            # 公式2
            self.h2 = np.dot(self.s1, self.Q)
        else:
            # 公式4
            self.h2 = np.dot(self.s1, self.Q) + np.dot(prev_s2, W2)
        # 公式6
        self.s2 = Tanh().forward(self.h2)

        if (isLast):
            # 公式7
            self.z = np.dot(self.s2, V)
            # 公式8
            self.a = self.z

    # for the first cell, prev_s should be zero
    # for the last cell, next_dh should be zero
    def backward(self, y, prev_s1, prev_s2, next_dh1, next_dh2, isFirst, isLast):
        if (isLast):
            # 公式10
            self.dz = (self.a - y)
        else:
            # 公式14
            self.dz = np.zeros_like(y)
        # end if

        if (isLast):
            # 公式12
            self.dh2 = np.dot(self.dz, self.V.T) * Tanh().backward(self.s2)
        else:
            # 公式15
            self.dh2 = np.dot(next_dh2, self.W2.T) * Tanh().backward(self.s2)
        # end if

        if (isLast):
            # 公式11
            self.dV = np.dot(self.s2.T, self.dz)
        else:
            # 公式16
            self.dV = np.zeros_like(self.V)
        
        # 公式21
        self.dQ = np.dot(self.s1.T, self.dh2)
        
        if (isLast):
            # 公式13
            self.dh1 = np.dot(self.dh2, self.Q.T) * Tanh().backward(self.s1)
        else:
            # 公式17
            self.dh1 = (np.dot(self.dh2, self.Q.T) + np.dot(next_dh1, self.W1.T)) * Tanh().backward(self.s1)

#             self.dh1 = np.dot(next_dh1, self.W1.T) * Tanh().backward(self.s1)

        # 公式22
        self.dU = np.dot(self.x.T, self.dh1)
        
        if (isFirst):
            # 公式18
            self.dW1 = np.zeros_like(self.W1)
            self.dW2 = np.zeros_like(self.W2)
        else:
            # 公式19,20
            self.dW1 = np.dot(prev_s1.T, self.dh1)
            self.dW2 = np.dot(prev_s2.T, self.dh2)
        # end if


class net(object):
    def __init__(self, hp, model_name):
        self.hp = hp
        self.model_name = model_name
        self.subfolder = os.getcwd() + "/" + self.__create_subfolder()
        print(self.subfolder)

        if (self.load_parameters(ParameterType.Init) == False):
            self.U,_ = WeightsBias_2_1.InitialParameters(self.hp.num_input, self.hp.num_hidden1, InitialMethod.Normal)
            self.Q,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden1, self.hp.num_hidden2, InitialMethod.Normal)
            self.V,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden2, self.hp.num_output, InitialMethod.Normal)
            self.W1,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden1, self.hp.num_hidden1, InitialMethod.Normal)
            self.W2,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden2, self.hp.num_hidden2, InitialMethod.Normal)
            self.save_parameters(ParameterType.Init)
        #end if

        self.loss_fun = LossFunction_1_1(self.hp.net_type)
        self.loss_trace = TrainingHistory_3_0()
        self.ts_list = []
        for i in range(self.hp.num_step+1):
            ts = timestep()
            self.ts_list.append(ts)
        #end for

    def __create_subfolder(self):
        if self.model_name != None:
            path = self.model_name.strip()
            path = path.rstrip("/")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def forward(self,X):
        self.x = X
        self.batch = self.x.shape[0]
        self.ts = self.x.shape[1]
        for i in range(0, self.ts):
            if (i == 0):
                self.ts_list[i].forward(
                    X[:,i], 
                    self.U, self.V, self.Q, self.W1, self.W2, 
                    None, None,
                   True, False)
            elif (i == self.ts - 1):
                self.ts_list[i].forward(
                    X[:,i], 
                    self.U, self.V, self.Q, self.W1, self.W2, 
                    self.ts_list[i-1].s1[0:self.batch], self.ts_list[i-1].s2[0:self.batch],
                    False, True)
            else:
                self.ts_list[i].forward(
                    X[:,i], 
                    self.U, self.V, self.Q, self.W1, self.W2, 
                    self.ts_list[i-1].s1[0:self.batch], self.ts_list[i-1].s2[0:self.batch],
                    False, False)
        #end for
        return self.ts_list[self.ts-1].a

    def backward(self,Y):
        for i in range(self.ts-1, -1, -1):
            if (i == 0):
                self.ts_list[i].backward(
                    Y, 
                    None, None,
                    self.ts_list[i+1].dh1[0:self.batch], self.ts_list[i+1].dh2[0:self.batch],
                    True, False)
            elif (i == self.ts - 1):
                self.ts_list[i].backward(
                    Y, 
                    self.ts_list[i-1].s1[0:self.batch], self.ts_list[i-1].s2[0:self.batch], 
                    None, None, 
                    False, True)
            else:
                self.ts_list[i].backward(
                    Y, 
                    self.ts_list[i-1].s1[0:self.batch], self.ts_list[i-1].s2[0:self.batch],
                    self.ts_list[i+1].dh1[0:self.batch], self.ts_list[i+1].dh2[0:self.batch], 
                    False, False)
        #end for

    def update(self):
        du = np.zeros_like(self.U)
        dv = np.zeros_like(self.V)
        dq = np.zeros_like(self.Q)
        dw1 = np.zeros_like(self.W1)
        dw2 = np.zeros_like(self.W2)
        for i in range(self.ts):
            du += self.ts_list[i].dU
            dv += self.ts_list[i].dV
            dq += self.ts_list[i].dQ
            dw1 += self.ts_list[i].dW1
            dw2 += self.ts_list[i].dW2
        #end for
        self.U = self.U - du * self.hp.eta / self.batch
        self.V = self.V - dv * self.hp.eta / self.batch
        self.Q = self.Q - dq * self.hp.eta / self.batch
        self.W1 = self.W1 - dw1 * self.hp.eta / self.batch
        self.W2 = self.W2 - dw2 * self.hp.eta / self.batch

    def save_parameters(self, para_type):
        if (para_type == ParameterType.Init):
            print("save init parameters...")
            self.file_name = str.format("{0}/init.npz", self.subfolder)
        elif (para_type == ParameterType.Best):
            print("save best parameters...")
            self.file_name = str.format("{0}/best.npz", self.subfolder)
        elif (para_type == ParameterType.Last):
            print("save last parameters...")
            self.file_name = str.format("{0}/last.npz", self.subfolder)
        #endif
        np.savez(self.file_name, U=self.U, V=self.V, Q=self.Q, W1=self.W1, W2=self.W2)

    def load_parameters(self, para_type):
        if (para_type == ParameterType.Init):
            print("load init parameters...")
            self.file_name = str.format("{0}/init.npz", self.subfolder)
            w_file = Path(self.file_name)
            if w_file.exists() is False:
                return False
        elif (para_type == ParameterType.Best):
            print("load best parameters...")
            self.file_name = str.format("{0}/best.npz", self.subfolder)
        elif (para_type == ParameterType.Last):
            print("load last parameters...")
            self.file_name = str.format("{0}/last.npz", self.subfolder)
        #endif
        data = np.load(self.file_name)
        self.U = data["U"]
        self.V = data["V"]
        self.Q = data["Q"]
        self.W1 = data["W1"]
        self.W2 = data["W2"]
        return True

    def train(self, dataReader, checkpoint=0.1):
        self.dataReader = dataReader
        min_loss = 10
        max_iteration = math.ceil(self.dataReader.num_train/self.hp.batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))
        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get data
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # forward
                self.forward(batch_x)
                # backward
                self.backward(batch_y)
                # update
                self.update()
                # check loss
                total_iteration = epoch * max_iteration + iteration               
                if (total_iteration+1) % checkpoint_iteration == 0:
                    #loss_train,acc_train = self.check_loss(batch_x, batch_y)
                    X,Y = dataReader.GetValidationSet()
                    loss_vld,acc_vld = self.check_loss(X,Y)
                    self.loss_trace.Add(epoch, total_iteration, None, None, loss_vld, acc_vld, None)
                    print(str.format("{0}:{1}:{2:3f} loss={3:6f}, acc={4:6f}", epoch, total_iteration, self.hp.eta, loss_vld, acc_vld))
                    if (loss_vld < min_loss):
                        min_loss = loss_vld
                        self.save_parameters(ParameterType.Best)
            #endif
        #end for
        self.save_parameters(ParameterType.Last)
        self.loss_trace.ShowLossHistory(
            self.hp.toString(),
            XCoordinate.Epoch)

    def check_loss(self,X,Y):
        a = self.forward(X)
        loss,acc = self.loss_fun.CheckLoss(a, Y)
        return loss, acc

    def test(self, dataReader):
        print("testing...")
        X,Y = dataReader.GetTestSet()
        count = X.shape[0]
        loss,acc = self.check_loss(X,Y)
        print(str.format("loss={0:6f}, acc={1:6f}", loss, acc)) 

def test(net, dataReader, num_step, pred_step, start, end):
    X,Y = dataReader.GetTestSet()
    assert(X.shape[0] == Y.shape[0])
    count = X.shape[0] - X.shape[0] % pred_step
    A = np.zeros((count,1))

    for i in range(0, count, pred_step):
        A[i:i+pred_step] = predict(net, X[i:i+pred_step], num_step, pred_step)

    loss,acc = net.loss_fun.CheckLoss(A, Y[0:count])
    print(str.format("pred_step={0}, loss={1:6f}, acc={2:6f}", pred_step, loss, acc))

    plt.plot(A[start+1:end+1], 'r-x', label="Pred")
    plt.plot(Y[start:end], 'b-o', label="True")
    plt.legend()
    plt.show()

def predict(net, X, num_step, pred_step):
    A = np.zeros((pred_step, 1))
    for i in range(pred_step):
        x = set_predicated_value(X[i:i+1], A, num_step, i)
        a = net.forward(x)
        A[i,0] = a
    #endfor
    return A

def set_predicated_value(X, A, num_step, predicated_step):
    x = X.copy()
    for i in range(predicated_step):
        x[0, num_step - predicated_step + i, 0] = A[i]
    #endfor
    return x

if __name__=='__main__':
    net_type = NetType.Fitting
    num_step = 24
    dataReader = load_data(net_type, num_step)
    eta = 0.05 # 0.1
    max_epoch = 100 # 100
    batch_size = 64 #64
    num_input = dataReader.num_feature
    num_hidden1 = 2
    num_hidden2 = 2
    num_output = dataReader.num_category

    model = str.format(
        "Level6_DeepRnn_{0}_{1}_{2}_{3}_{4}_{5}_{6}", 
        max_epoch, batch_size, num_input, num_hidden1, num_hidden2, num_output, eta)
    hp = HyperParameters_4_4(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden1, num_hidden2, num_output, 
        net_type)

    n = net(hp, model)
    #n.train(dataReader, checkpoint=1)

    n.load_parameters(ParameterType.Last)
    pred_steps = [8,4,2,1]
    for i in range(4):
        test(n, dataReader, num_step, pred_steps[i], 1050, 1150)

    n.load_parameters(ParameterType.Best)
    pred_steps = [8,4,2,1]
    for i in range(4):
        test(n, dataReader, num_step, pred_steps[i], 1050, 1150)
