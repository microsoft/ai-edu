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
    def Hook(self):
        self.GetWeightBias()

    def GetWeightBias(self):
        layer = self.layer_list[0]
        w_history.append(layer.wb.W[0,0])
        b_history.append(layer.wb.B[0,0])


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

def ShowContour(net, dataReader):
    last_w = w_history[-1]
    last_b = b_history[-1]
    X,Y=dataReader.GetTestSet()
    len1 = 50
    len2 = 50
    w = np.linspace(last_w-1,last_w+1,len1)
    b = np.linspace(last_b-1,last_b+1,len2)
    W,B = np.meshgrid(w,b)
    len3 = len1 * len2
    m = X.shape[0]
    Z = np.dot(X, W.ravel().reshape(1,len3)) + B.ravel().reshape(1,len3)
    Loss1 = (Z - Y)**2
    Loss2 = Loss1.sum(axis=0,keepdims=True)/m
    Loss3 = Loss2.reshape(len1, len2)
    plt.contour(W,B,Loss3,levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)

    # show w,b trace
    plt.plot(w_history,b_history)
    plt.xlabel("w")
    plt.ylabel("b")
    title = str.format("lr:{0},op:{1},it:{2},w={3:.3f},b={4:.3f}", net.hp.eta, net.hp.optimizer_name.name, len(w_history), last_w, last_b)
    plt.title(title)

    plt.axis([last_w-1,last_w+1,last_b-1,last_b+1])
    plt.show()

def model(learning_rate, optimizer):
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    hp = HyperParameters_4_1(
        eta=learning_rate, max_epoch=100, batch_size=5,
        net_type=NetType.Fitting, 
        init_method=InitialMethod.Zero,
        optimizer_name=optimizer,
        stopper=Stopper(StopCondition.StopLoss, 0.02))

    net = NeuralNet_4_2(hp, "linear_regression")
    fc1 = FcLayer_1_1(1,1,hp)
    net.add_layer(fc1, "fc1")

    net.train(dr, checkpoint=0.1, need_test=False)
    #net.ShowLossHistory()
    #ShowResult(net, dr)
    ShowContour(net, dr)

if __name__ == '__main__':

    dict = {
            OptimizerName.SGD:0.5, 
            OptimizerName.Momentum:0.05, 
            OptimizerName.Nag:0.05,
            OptimizerName.AdaGrad:1,
            OptimizerName.AdaDelta:0.0,
            OptimizerName.RMSProp:0.1, 
            OptimizerName.Adam:0.5
            }

    w_history = []
    b_history = []

    for key in dict.keys():
        model(dict[key], key)
        w_history.clear()
        b_history.clear()
    