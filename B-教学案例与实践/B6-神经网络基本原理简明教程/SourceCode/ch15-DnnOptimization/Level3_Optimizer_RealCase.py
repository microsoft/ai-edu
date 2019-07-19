# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# multiple iteration, loss calculation, stop condition, predication
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from MiniFramework.NeuralNet_4_1 import *

train_file = "../../data/ch04.npz"
test_file = "../../data/ch04.npz"

class NeuralNet_4_2(NeuralNet_4_1):
    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):

        self.GetWeightBias()

        print("epoch=%d, total_iteration=%d" %(epoch, total_iteration))

        # calculate train loss
        self.forward(train_x, train=False)
        loss_train = self.lossFunc.CheckLoss(self.output, train_y)
        loss_train = loss_train# + regular_cost / train_x.shape[0]
        accuracy_train = self.CalAccuracy(self.output, train_y)
        print("loss_train=%.6f, accuracy_train=%f" %(loss_train, accuracy_train))

        # calculate validation loss
        vld_x, vld_y = dataReader.GetValidationSet()
        self.forward(vld_x, train=False)
        loss_vld = self.lossFunc.CheckLoss(self.output, vld_y)
        loss_vld = loss_vld #+ regular_cost / vld_x.shape[0]
        accuracy_vld = self.CalAccuracy(self.output, vld_y)
        print("loss_valid=%.6f, accuracy_valid=%f" %(loss_vld, accuracy_vld))

        need_stop = self.loss_trace.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld, self.hp.eps)
        if loss_vld <= self.hp.eps:
            need_stop = True
        return need_stop

    def GetWeightBias(self):
        layer = self.layer_list[0]
        w_history.append(layer.weights.W[0,0])
        b_history.append(layer.weights.B[0,0])


def ShowResult(net, dataReader):
    X,Y = dataReader.GetTestSet()
    # draw sample data
    plt.plot(X, Y, "b.")
    # draw predication data
    PX = np.linspace(0,1,5).reshape(5,1)
    PZ = net.inference(PX)
    plt.plot(PX, PZ, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()

def ShowContour():
    plt.plot(w_history, b_history)
    plt.show()

if __name__ == '__main__':

    w_history = []
    b_history = []

    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    hp = HyperParameters_4_0(
        eta=0.3, max_epoch=100, batch_size=10, eps=1e-6,
        net_type=NetType.Fitting, 
        init_method=InitialMethod.Normal,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(hp, "linear_regression")
    fc1 = FcLayer_1_1(1,1,hp)
    net.add_layer(fc1, "fc1")

    net.train(dr, checkpoint=1, need_test=False)
    #net.ShowLossHistory()
    #ShowResult(net, dr)
    ShowContour()

