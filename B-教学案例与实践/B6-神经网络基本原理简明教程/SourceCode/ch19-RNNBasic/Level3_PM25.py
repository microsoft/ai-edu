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
    dr.GenerateValidationSet(k=100)
    return dr

class timestep(object):
    # for the first cell, prev_s should be zero
    def forward(self, x, U, V, W, prev_s, isFirst, isLast):
        self.U = U
        self.V = V
        self.W = W
        self.x = x

        if (isFirst):
            # 公式1
            self.h = np.dot(x, U)
        else:
            # 公式2
            self.h = np.dot(x, U) + np.dot(prev_s, W) 
        #endif

        # 公式3
        self.s = Tanh().forward(self.h)

        if (isLast):
            # 公式4
            self.z = np.dot(self.s, V)
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
        else:
            self.dz = np.zeros_like(y)
            # 公式9
            self.dh = np.dot(next_dh, self.W.T) * Tanh().backward(self.s)
            self.dV = np.zeros_like(self.V)
        #endif

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

        self.zero_state = np.zeros((self.hp.batch_size, self.hp.num_hidden))
        self.loss_fun = LossFunction_1_1(self.hp.net_type)
        self.loss_trace = TrainingHistory_3_0()
        self.ts_list = []
        for i in range(self.hp.num_step+1): # create one more ts to hold zero values
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
                self.ts_list[i].forward(X[:,i], self.U, self.V, self.W, None, True, False)
            elif (i == self.ts - 1):
                self.ts_list[i].forward(X[:,i], self.U, self.V, self.W, self.ts_list[i-1].s[0:self.batch], False, True)
            else:
                self.ts_list[i].forward(X[:,i], self.U, self.V, self.W, self.ts_list[i-1].s[0:self.batch], False, False)
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

    def update(self):
        du = np.zeros_like(self.U)
        dv = np.zeros_like(self.V)
        dw = np.zeros_like(self.W)
        for i in range(self.ts):
            du += self.ts_list[i].dU
            dv += self.ts_list[i].dV
            dw += self.ts_list[i].dW
        #end for
        self.U = self.U - du * self.hp.eta
        self.V = self.V - dv * self.hp.eta
        self.W = self.W - dw * self.hp.eta

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

        for epoch in range(self.hp.max_epoch):
            #self.hp.eta = self.lr_decay(epoch)
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
                    X,Y = dataReader.GetValidationSet()
                    loss,acc = self.check_loss(X,Y)
                    self.loss_trace.Add(epoch, total_iteration, None, None, loss, acc, None)
                    print(str.format("{0}:{1}:{2} loss={3:6f}, acc={4:6f}", epoch, total_iteration, self.hp.eta, loss, acc))
                    if (loss < min_loss):
                        min_loss = loss
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


    def lr_decay(self, epoch):
        if (epoch < 20):
            return 0.001
        elif (epoch < 40):
            return 0.0005
        elif (epoch < 60):
            return 0.0002
        elif (epoch < 80):
            return 0.0001
        else:
            return 0.00005

    def test(self, dataReader):
        print("testing...")
        X,Y = dataReader.GetTestSet()
        count = X.shape[0]
        loss,acc = self.check_loss(X,Y)
        print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))

    def draw_confusion_matrix(self, dataReader, confusion_matrix):
        for i in range(dataReader.num_category):
            confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion_matrix)
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + dataReader.language_list, rotation=90)
        ax.set_yticklabels([''] + dataReader.language_list)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # sphinx_gallery_thumbnail_number = 2
        plt.show()


if __name__=='__main__':
    net_type = NetType.MultipleClassifier
    num_step = 12 #24
    dataReader = load_data(net_type, num_step)
    eta = 0.0001   # 0.0001
    max_epoch = 200
    batch_size = 128
    num_input = dataReader.num_feature
    num_hidden = 32  # 16
    num_output = dataReader.num_category
    model = str.format("Level3_{0}_{1}_{2}_{3}", max_epoch, batch_size, num_hidden, eta)
    hp = HyperParameters_4_3(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden, num_output, 
        net_type)
    n = net(hp, model)
    #n.load_parameters()
    n.train(dataReader, checkpoint=1)
