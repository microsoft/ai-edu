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
from ExtendedDataReader.NameDataReader import *

file = "../../data/ch19.name_language.txt"

def load_data():
    dr = NameDataReader()
    dr.ReadData(file)
    dr.GenerateValidationSet(1000)
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
        else:
            self.dz = np.zeros_like(y)
        # end if
        if (isLast):
            # 公式8
            self.dh = np.dot(self.dz, self.V.T) * Tanh().backward(self.s)
        else:
            # 公式9
            self.dh = np.dot(next_dh, self.W.T) * Tanh().backward(self.s)
        # end if
        if (isLast):
            # 公式10
            self.dV = np.dot(self.s.T, self.dz)
        else:
            self.dV = np.zeros_like(self.V)
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

        if (self.load_init_value() == False):
            self.U,_ = WeightsBias_2_1.InitialParameters(self.hp.num_input, self.hp.num_hidden, InitialMethod.Normal)
            self.V,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden, self.hp.num_output, InitialMethod.Normal)
            self.W,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden, self.hp.num_hidden, InitialMethod.Normal)
            self.save_init_value()
        #end if

        self.zero_state = np.zeros((self.hp.batch_size, self.hp.num_hidden))
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
        self.U = self.U - du * self.hp.eta / self.batch
        self.V = self.V - dv * self.hp.eta / self.batch
        self.W = self.W - dw * self.hp.eta / self.batch

    def save_init_value(self):
        self.init_file_name = str.format("{0}/init.npz", self.subfolder)
        np.savez(self.init_file_name, U=self.U, V=self.V, W=self.W)

    def load_init_value(self):
        self.init_file_name = str.format("{0}/init.npz", self.subfolder)
        w_file = Path(self.init_file_name)
        if w_file.exists():
            data = np.load(self.init_file_name)
            self.U = data["U"]
            self.V = data["V"]
            self.W = data["W"]
            return True
        else:
            return False

    def save_parameters(self):
        print("save best parameters...")
        self.result_file_name = str.format("{0}/result.npz", self.subfolder)
        np.savez(self.result_file_name, U=self.U, V=self.V, W=self.W)

    def load_parameters(self):
        print("load best parameters...")
        self.result_file_name = str.format("{0}/result.npz", self.subfolder)
        data = np.load(self.result_file_name)
        self.U = data["U"]
        self.V = data["V"]
        self.W = data["W"]

    def check_loss(self,X,Y):
        LOSS = 0
        ACC = 0
        for i in range(self.dataReader.num_dev):
            a = self.forward(X[i])
            loss,acc = self.loss_fun.CheckLoss(a, Y[i])
            LOSS += loss
            ACC += acc
        return LOSS/self.dataReader.num_dev, ACC/self.dataReader.num_dev

    def train(self, dataReader, checkpoint=0.1):
        self.dataReader = dataReader
        min_loss = 10
        total_iter = 0
        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            while(True):
                # get data
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size)
                if (batch_x is None):
                    break
                # forward
                self.forward(batch_x)
                # backward
                self.backward(batch_y)
                # update
                self.update()
                total_iter += 1
            #enf while
            # check loss
            X,Y = dataReader.GetValidationSet()
            loss,acc = self.check_loss(X,Y)
            self.loss_trace.Add(epoch, total_iter, None, None, loss, acc, None)
            print(str.format("{0}:{1}:{2} loss={3:6f}, acc={4:6f}", epoch, total_iter, self.hp.eta, loss, acc))
            if (loss < min_loss):
                min_loss = loss
                self.save_parameters()
            #endif
        #end for
        self.loss_trace.ShowLossHistory("Loss and Accuracy", XCoordinate.Epoch)

    def test(self, dataReader):
        confusion_matrix = np.zeros((dataReader.num_category, dataReader.num_category))
        correct = 0
        for i in range(dataReader.num_train):
            x,y = dataReader.GetBatchTrainSamples(1)
            output = self.forward(x)
            pred = np.argmax(output)
            label = np.argmax(y)
            confusion_matrix[label, pred] += 1
            if (pred == label):
                correct += 1
        #end for
        print(str.format("correctness={0}/{1}={2}", correct, dataReader.num_train, correct / dataReader.num_train))
        self.draw_confusion_matrix(dataReader, confusion_matrix)

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
    dataReader = load_data()
    eta = 0.02 # 0.02
    max_epoch = 100 # 100
    batch_size = 8 # 8
    num_input = dataReader.num_feature
    num_hidden = 16 # 16
    num_output = dataReader.num_category
    model = str.format("Level5_{0}_{1}_{2}_{3}_{4}_{5}", max_epoch, batch_size, num_input, num_hidden, num_output, eta)
    hp = HyperParameters_4_3(
        eta, max_epoch, batch_size, 
        dataReader.max_step, num_input, num_hidden, num_output, 
        OutputType.LastStep, NetType.MultipleClassifier)
    n = net(hp, model)
    n.train(dataReader, checkpoint=1)

    # last 
    n.test(dataReader)
    # best
    n.load_parameters()
    dataReader.ResetPointer()
    n.test(dataReader)
