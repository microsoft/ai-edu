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

class timestep_classification(object):
    # for the first cell, prev_s should be zero
    def forward(self, x, U, bu, V, bv, W, prev_s, isFirst, isLast):
        self.U = U
        self.bu = bu
        self.V = V
        self.bv = bv
        self.W = W
        self.x = x

        if (isFirst):
            # 公式1
            self.h = np.dot(x, U) + self.bu
        else:
            # 公式2
            self.h = np.dot(x, U) + np.dot(prev_s, W) + self.bu
        #endif

        # 公式3
        self.s = Tanh().forward(self.h)

        if (isLast):
            # 公式4
            self.z = np.dot(self.s, V) + self.bv
            # 公式5
            self.a = Softmax().forward(self.z)

    # for the first cell, prev_s should be zero
    # for the last cell, next_dh should be zero
    def backward(self, y, prev_s, next_dh, isFirst, isLast):
        if (isLast):
            # 公式7
            self.dz = (self.a - y)
            # 公式8
            self.dh = np.dot(self.dz, self.V.T) * Tanh().backward(self.s)
            # 公式10
            self.dV = np.dot(self.s.T, self.dz)
            self.dbv = self.dz
        else:
            self.dz = np.zeros_like(y)
            self.dbv = np.zeros_like(self.bv)
            # 公式9
            self.dh = np.dot(next_dh, self.W.T) * Tanh().backward(self.s)
            self.dV = np.zeros_like(self.V)
        #endif
        self.dbv = np.sum(self.dz, axis=0, keepdims=True)/y.shape[0]
        self.dbu = np.sum(self.dh, axis=0, keepdims=True)/y.shape[0]

        # 公式11
        self.dU = np.dot(self.x.T, self.dh)

        if (isFirst):
            self.dW = np.zeros_like(self.W)
        else:
            # 公式12
            self.dW = np.dot(prev_s.T, self.dh)
        # end if

class timestep_fit(object):
    # for the first cell, prev_s should be zero
    def forward(self, x, U, bu, V, bv, W, prev_s, isFirst, isLast):
        self.U = U
        self.bu = bu
        self.V = V
        self.bv = bv
        self.W = W
        self.x = x

        if (isFirst):
            # 公式1
            self.h = np.dot(x, U) + self.bu
        else:
            # 公式2
            self.h = np.dot(x, U) + np.dot(prev_s, W) + self.bu
        #endif

        # 公式3
        self.s = Tanh().forward(self.h)

        if (isLast):
            # 公式4
            self.z = np.dot(self.s, V) + self.bv
            self.a = self.z

    # for the first cell, prev_s should be zero
    # for the last cell, next_dh should be zero
    def backward(self, y, prev_s, next_dh, isFirst, isLast):
        if (isLast):
            # 公式7
            self.dz = (self.z - y)
            # 公式8
            self.dh = np.dot(self.dz, self.V.T) * Tanh().backward(self.s)
            # 公式10
            self.dV = np.dot(self.s.T, self.dz)
            self.dbv = self.dz
        else:
            self.dz = np.zeros_like(y)
            self.dbv = np.zeros_like(self.bv)
            # 公式9
            self.dh = np.dot(next_dh, self.W.T) * Tanh().backward(self.s)
            self.dV = np.zeros_like(self.V)
        #endif
        #self.dbv = np.sum(self.dz, axis=0, keepdims=True)/y.shape[0]
        #self.dbu = np.sum(self.dh, axis=0, keepdims=True)/y.shape[0]

        # 公式11
        self.dU = np.dot(self.x.T, self.dh)

        if (isFirst):
            self.dW = np.zeros_like(self.W)
        else:
            # 公式12
            self.dW = np.dot(prev_s.T, self.dh)
        # end if

class net(object):
    def __init__(self, hp, model_name):
        self.hp = hp
        self.model_name = model_name
        self.subfolder = os.getcwd() + "/" + self.__create_subfolder()
        print(self.subfolder)

        if (self.load_parameters(ParameterType.Init) == False):
            self.U,_ = WeightsBias_2_1.InitialParameters(self.hp.num_input, self.hp.num_hidden, InitialMethod.Normal)
            self.V,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden, self.hp.num_output, InitialMethod.Normal)
            self.W,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden, self.hp.num_hidden, InitialMethod.Normal)
            self.save_parameters(ParameterType.Init)
        #end if
        self.bu = np.zeros((1, self.hp.num_hidden))
        self.bv = np.zeros((1, self.hp.num_output))

        self.zero_state = np.zeros((self.hp.batch_size, self.hp.num_hidden))
        self.loss_fun = LossFunction_1_1(self.hp.net_type)
        self.loss_trace = TrainingHistory_3_0()
        self.ts_list = []
        for i in range(self.hp.num_step+1): # create one more ts to hold zero values
            #ts = timestep_fit()
            ts = timestep_classification()
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
                self.ts_list[i].forward(X[:,i], self.U, self.bu, self.V, self.bv, self.W, None, True, False)
            elif (i == self.ts - 1):
                self.ts_list[i].forward(X[:,i], self.U, self.bu, self.V, self.bv, self.W, self.ts_list[i-1].s[0:self.batch], False, True)
            else:
                self.ts_list[i].forward(X[:,i], self.U, self.bu, self.V, self.bv, self.W, self.ts_list[i-1].s[0:self.batch], False, False)
        #end for
        return self.ts_list[self.ts-1].a

    def backward(self,Y):
        for i in range(self.ts-1, -1, -1):
            if (i == 0):
                self.ts_list[i].backward(Y, None, self.ts_list[i+1].dh[0:self.batch], True, False)
            elif (i == self.ts - 1):
                self.ts_list[i].backward(Y, self.ts_list[i-1].s[0:self.batch], None, False, True)
            else:
                self.ts_list[i].backward(Y, self.ts_list[i-1].s[0:self.batch], self.ts_list[i+1].dh[0:self.batch], False, False)
        #end for

    def update(self, batch_size):
        du = np.zeros_like(self.U)
        #dbu = np.zeros_like(self.bu)
        dv = np.zeros_like(self.V)
        #dbv = np.zeros_like(self.bv)
        dw = np.zeros_like(self.W)
        for i in range(self.ts):
            du += self.ts_list[i].dU
            #dbu += self.ts_list[i].dbu
            dv += self.ts_list[i].dV
            #dbv += self.ts_list[i].dbv
            dw += self.ts_list[i].dW
        #end for
        self.U = self.U - du * self.hp.eta / batch_size
        #self.bu = self.bu - dbu * self.hp.eta
        self.V = self.V - dv * self.hp.eta / batch_size
        #self.bv = self.bv - dbv * self.hp.eta
        self.W = self.W - dw * self.hp.eta / batch_size

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
        np.savez(self.file_name, U=self.U, V=self.V, W=self.W)

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
        self.W = data["W"]
        return True

    def check_loss(self,X,Y):
        a = self.forward(X)
        loss,acc = self.loss_fun.CheckLoss(a, Y)
        return loss, acc

    def train(self, dataReader, checkpoint=0.1):
        self.dataReader = dataReader
        min_loss = 10
        max_iteration = math.ceil(self.dataReader.num_train/batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))
        lr_start = self.hp.eta
        decay = 0.01
        for epoch in range(self.hp.max_epoch):
            self.hp.eta = self.lr_decay(lr_start, decay, epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get data
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # forward
                self.forward(batch_x)
                # backward
                self.backward(batch_y)
                # update
                self.update(batch_x.shape[0])
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
        self.test(self.dataReader)
        self.load_parameters(ParameterType.Best)
        self.test(self.dataReader)
        self.loss_trace.ShowLossHistory(
            str.format("epoch:{0},batch:{1},hidden:{2},eta:{3}", max_epoch, batch_size, num_hidden, eta), 
            XCoordinate.Epoch)


    def lr_decay(self, lr_start, decay, epoch):
        #lr = lr_start / (1.0 + decay * epoch)
        #return lr
        if (epoch < 20):
            return 0.1
        elif (epoch < 50):
            return 0.05
        elif (epoch < 80):
            return 0.01
        elif (epoch < 100):
            return 0.005
        else:
            return 0.001

    def test(self, dataReader):
        print("testing...")
        X,Y = dataReader.GetTestSet()
        count = X.shape[0]
        loss,acc = self.check_loss(X,Y)
        print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))
        A = self.forward(X)
        ra = np.argmax(A, axis=1)
        ry = np.argmax(Y, axis=1)
        p1, = plt.plot(ra[0:200])
        p2, = plt.plot(ry[0:200])
        plt.legend([p1,p2], ["pred","true"])
        plt.show()

        p1, = plt.plot(ra[1000:1200])
        p2, = plt.plot(ry[1000:1200])
        plt.legend([p1,p2], ["pred","true"])
        plt.show()



if __name__=='__main__':
    net_type = NetType.MultipleClassifier
    num_step = 8 #8
    dataReader = load_data(net_type, num_step)
    eta = 0.1   
    max_epoch = 100
    batch_size = 64 #64
    num_input = dataReader.num_feature
    num_hidden = 16  # 16
    num_output = dataReader.num_category
    model = str.format("Level3_{0}_{1}_{2}_{3}", max_epoch, batch_size, num_hidden, eta)
    hp = HyperParameters_4_3(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden, num_output, 
        net_type)
    n = net(hp, model)
    #n.load_parameters()
    n.train(dataReader, checkpoint=1)
