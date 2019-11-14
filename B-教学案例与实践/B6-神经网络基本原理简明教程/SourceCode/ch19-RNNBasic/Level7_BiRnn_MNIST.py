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
from ExtendedDataReader.MnistImageDataReader import *

def load_data():
    dataReader = MnistImageDataReader(mode="timestep")
    #dataReader.ReadLessData(10000)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=12)
    return dataReader

class timestep(object):
    def forward_1(self, x1, U1, bU1, W1, prev_s1, isFirst):
        self.x1 = x1
        self.U1 = U1
        self.bU1 = bU1
        self.W1 = W1

        if (isFirst):
            # 公式1
            self.h1 = np.dot(x1, U1) + self.bU1
        else:
            # 公式1
            self.h1 = np.dot(x1, U1) + np.dot(prev_s1, W1)  + self.bU1
        #endif
        # 公式2
        self.s1 = Tanh().forward(self.h1)

    def forward_2(self, x2, U2, bU2, W2, prev_s2, isFirst):
        self.x2 = x2
        self.U2 = U2
        self.bU2 = bU2
        self.W2 = W2
        if (isFirst):
            # 公式3
            self.h2 = np.dot(self.x2, self.U2) + self.bU2
        else:
            # 公式3
            self.h2 = np.dot(self.x2, self.U2) + np.dot(prev_s2, self.W2) + self.bU2
        #endif
        # 公式4
        self.s2 = Tanh().forward(self.h2)

    def forward_3(self, V, bV, isLast):
        self.V = V
        self.bV = bV
        self.s = self.s1 + self.s2
        if (isLast):
            # 公式6
            self.z = np.dot(self.s, self.V) + self.bV
            # 公式7
            self.a = Softmax().forward(self.z)

    def backward_3(self, y, isFirst, isLast):
        if (isLast):
            self.dz = self.a - y
        else:
            self.dz = np.zeros_like(y)
#        self.dz = self.a - y
        self.dbV = np.sum(self.dz, axis=0, keepdims=True)
        # 公式11
        self.dV = np.dot(self.s.T, self.dz)

    def backward_1(self, prev_s1, next_dh1, isFirst, isLast):
        if (isLast):
            # 公式9
            self.dh1 = np.dot(self.dz, self.V.T) * Tanh().backward(self.s1)
        else:
            # 公式10
            self.dh1 = (np.dot(self.dz, self.V.T) + np.dot(next_dh1, self.W1.T)) * Tanh().backward(self.s1)

        self.dbU1 = np.sum(self.dh1, axis=0, keepdims=True)

        # 公式12
        self.dU1 = np.dot(self.x1.T, self.dh1)
        
        if (isFirst):
            # 公式14
            self.dW1 = np.zeros_like(self.W1)
        else:
            # 公式13
            self.dW1 = np.dot(prev_s1.T, self.dh1)
        # end if

    def backward_2(self, prev_s2, next_dh2, isFirst, isLast):
        if (isLast):
            # 公式9
            self.dh2 = np.dot(self.dz, self.V.T) * Tanh().backward(self.s2)
        else:
            # 公式10
            self.dh2 = (np.dot(self.dz, self.V.T) + np.dot(next_dh2, self.W2.T)) * Tanh().backward(self.s2)

        self.dbU2 = np.sum(self.dh2, axis=0, keepdims=True)

        # 公式12
        self.dU2 = np.dot(self.x2.T, self.dh2)
        
        if (isFirst):
            # 公式14
            self.dW2 = np.zeros_like(self.W2)
        else:
            # 公式13
            self.dW2 = np.dot(prev_s2.T, self.dh2)
        # end if

class net(object):
    def __init__(self, hp, model_name):
        self.hp = hp
        self.model_name = model_name
        self.subfolder = os.getcwd() + "/" + self.__create_subfolder()
        print(self.subfolder)
        assert(self.hp.num_hidden1 == self.hp.num_hidden2)
        if (self.load_parameters(ParameterType.Init) == False):
            self.U1,self.bU1 = WeightsBias_2_1.InitialParameters(self.hp.num_input, self.hp.num_hidden1, InitialMethod.Normal)
            self.U2,self.bU2 = WeightsBias_2_1.InitialParameters(self.hp.num_input, self.hp.num_hidden2, InitialMethod.Normal)
            self.V,self.bV = WeightsBias_2_1.InitialParameters(self.hp.num_hidden1, self.hp.num_output, InitialMethod.Normal)
            self.W1,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden1, self.hp.num_hidden1, InitialMethod.Normal)
            self.W2,_ = WeightsBias_2_1.InitialParameters(self.hp.num_hidden2, self.hp.num_hidden2, InitialMethod.Normal)
            self.save_parameters(ParameterType.Init)
        #end if

        self.loss_fun = LossFunction_1_1(self.hp.net_type)
        self.loss_trace = TrainingHistory_3_0()
        self.ts_list = []
        for i in range(self.hp.num_step):
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

        # 1
        for i in range(0, self.ts):
            if (i == 0):
                prev_s1 = None
                isFirst = True
            else:
                prev_s1 = self.ts_list[i-1].s1
                isFirst = False
            #endif
            self.ts_list[i].forward_1(X[:,i], self.U1, self.bU1, self.W1, prev_s1, isFirst)
        #endfor
        # 2
        for i in range(0, self.ts):
            if (i == 0):
                isFirst = True
                prev_s2 = None
            else:
                isFirst = False
                prev_s2 = self.ts_list[i-1].s2
            #endif
            self.ts_list[i].forward_2(X[:,self.ts-i-1], self.U2, self.bU2, self.W2, prev_s2, isFirst)
        #end for
        # sum
        for i in range(0, self.ts):
            if (i == self.ts - 1):
                isLast = True
            else:
                isLast = False
            #endif
            self.ts_list[i].forward_3(self.V, self.bV, isLast)
        #endfor
        return self.ts_list[self.ts-1].a

    def backward(self, Y):
        # sum
        for i in range(0, self.ts):
            if (i == 0):
                isFirst = True
                isLast = False
            elif (i == self.ts-1):
                isFirst = False
                isLast = True
            else:
                isFirst = False
                isLast = False
            #endif
            self.ts_list[i].backward_3(Y, isFirst, isLast)
        # 1
        for i in range(self.ts-1, -1, -1):
            if (i == self.ts - 1):
                next_dh1 = None
                prev_s1 = self.ts_list[i-1].s1
                isLast = True
                isFirst = False
            elif (i == 0):
                next_dh1 = self.ts_list[i+1].dh1
                prev_s1 = None
                isLast = False
                isFirst = True
            else:
                next_dh1 = self.ts_list[i+1].dh1
                prev_s1 = self.ts_list[i-1].s1
                isLast = False
                isFirst = False
            #endif
            self.ts_list[i].backward_1(prev_s1, next_dh1, isFirst, isLast)
        #end for
        #2
        for i in range(self.ts-1, -1, -1):
            if (i == self.ts - 1):
                next_dh2 = None
                prev_s2 = self.ts_list[i-1].s2
                isLast = True
                isFirst = False
            elif (i == 0):
                next_dh2 = self.ts_list[i+1].dh2
                prev_s2 = None
                isLast = False
                isFirst = True
            else:
                next_dh2 = self.ts_list[i+1].dh2
                prev_s2 = self.ts_list[i-1].s2
                isLast = False
                isFirst = False
            #endif
            self.ts_list[i].backward_2(prev_s2, next_dh2, isFirst, isLast)
        #end for
        #end for

    def update(self):
        dU1 = np.zeros_like(self.U1)
        dbU1 = np.zeros_like(self.bU1)
        dU2 = np.zeros_like(self.U2)
        dbU2 = np.zeros_like(self.bU2)
        dV = np.zeros_like(self.V)
        dbV = np.zeros_like(self.bV)
        dW1 = np.zeros_like(self.W1)
        dW2 = np.zeros_like(self.W2)
        for i in range(self.ts):
            dU1 += self.ts_list[i].dU1
            dbU1 += self.ts_list[i].dbU1
            dU2 += self.ts_list[i].dU2
            dbU2 += self.ts_list[i].dbU2
            dV  += self.ts_list[i].dV
            dbV  += self.ts_list[i].dbV
            dW1 += self.ts_list[i].dW1
            dW2 += self.ts_list[i].dW2
        #end for
        self.U1 = self.U1 - dU1 * self.hp.eta / self.batch
        self.bU1 = self.bU1 - dbU1 * self.hp.eta / self.batch
        self.U2 = self.U2 - dU2 * self.hp.eta / self.batch
        self.bU2 = self.bU2 - dbU2 * self.hp.eta / self.batch
        self.V  = self.V  - dV  * self.hp.eta / self.batch
        self.bV  = self.bV - dbV  * self.hp.eta / self.batch
        self.W1 = self.W1 - dW1 * self.hp.eta / self.batch
        self.W2 = self.W2 - dW2 * self.hp.eta / self.batch

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
        np.savez(self.file_name, U1=self.U1, bU1=self.bU1, U2=self.U2, bU2=self.bU2, V=self.V, bV=self.bV, W1=self.W1, W2=self.W2)

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
        self.U1 = data["U1"]
        self.bU1 = data["bU1"]
        self.U2 = data["U2"]
        self.bU2 = data["bU2"]
        self.V  = data["V"]
        self.bV  = data["bV"]
        self.W1 = data["W1"]
        self.W2 = data["W2"]
        return True

    def check_loss(self,X,Y):
        a = self.forward(X)
        loss,acc = self.loss_fun.CheckLoss(a, Y)
        return loss, acc

    def learning_rate_decay(self, epoch):
        if (epoch < 30):
            return self.hp.eta
        elif (epoch < 50):
            return 0.008
        elif (epoch < 70):
            return 0.005
        else:
            return 0.002

    def train(self, dataReader, checkpoint=0.1):
        self.dataReader = dataReader
        min_loss = 10
        max_iteration = math.ceil(self.dataReader.num_train/self.hp.batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))
        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            self.hp.eta = self.learning_rate_decay(epoch)
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
                    loss_train,acc_train = self.check_loss(batch_x, batch_y)
                    X,Y = dataReader.GetValidationSet()
                    loss_vld,acc_vld = self.check_loss(X,Y)
                    self.loss_trace.Add(epoch, total_iteration, loss_train, acc_train, loss_vld, acc_vld, None)
                    print(str.format("{0}:{1}:{2:3f} loss={3:6f}, acc={4:6f}", epoch, total_iteration, self.hp.eta, loss_vld, acc_vld))
                    if (loss_vld < min_loss):
                        min_loss = loss_vld
                        self.save_parameters(ParameterType.Best)
            #endif
        #end for
        self.save_parameters(ParameterType.Last)
        self.loss_trace.ShowLossHistory(self.hp.toString(), XCoordinate.Iteration)

    def test(self, dataReader):
        print("testing...")
        X,Y = dataReader.GetTestSet()
        count = X.shape[0]
        loss,acc = self.check_loss(X,Y)
        print(str.format("loss={0:6f}, acc={1:6f}", loss, acc)) 
    
if __name__=='__main__':
    dataReader = load_data()
    eta = 0.01
    max_epoch = 100
    batch_size = 128
    num_step = 28
    num_input = dataReader.num_feature
    num_hidden1 = 20
    num_hidden2 = 20
    num_output = dataReader.num_category
    model = str.format(
        "Level7_BiRNN_{0}_{1}_{2}_{3}_{4}_{5}_{6}",                        
        max_epoch, batch_size, num_input, num_hidden1, num_hidden2, num_output, eta)
    hp = HyperParameters_4_4(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden1, num_hidden2, num_output, 
        NetType.MultipleClassifier)
    n = net(hp, model)
    n.train(dataReader, checkpoint=0.5)
    n.test(dataReader)# last
    n.load_parameters(ParameterType.Best)
    n.test(dataReader)# best
