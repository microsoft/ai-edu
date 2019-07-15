# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet41 import *
from MiniFramework.ActivatorLayer import *

train_file = "../../Data/ch16.train.npz"
test_file = "../../Data/ch16.test.npz"


def Model(dataReader, num_input, num_hidden, num_output, params):
    net = NeuralNet41(params, "overfitting")

    fc1 = FcLayer(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    s1 = ActivatorLayer(Sigmoid())
    net.add_layer(s1, "s1")
    
    fc2 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc2, "fc2")
    relu2 = ActivatorLayer(Sigmoid())
    net.add_layer(relu2, "relu2")
    
    fc3 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc3, "fc3")
    relu3 = ActivatorLayer(Sigmoid())
    net.add_layer(relu3, "relu3")
    
    fc4 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc4, "fc4")
    relu4 = ActivatorLayer(Relu())
    net.add_layer(relu4, "relu4")
    
    fc5 = FcLayer(num_hidden, num_output, params)
    net.add_layer(fc5, "fc5")

    net.train(dataReader, checkpoint=100, need_test=True)
    net.ShowLossHistory(XCoordinate.Epoch)
    
    return net

def ShowResult(net, dr):
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, c='red')
    plt.title("fitting result")
    plt.scatter(dr.XTrain, dr.YTrain)
    plt.scatter(dr.XTest, dr.YTest, marker='x')
    plt.show()

def LoadData():
    dr = DataReader20(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(NetType.Fitting)
   # dr.Shuffle()
    return dr

if __name__ == '__main__':

    dr = LoadData()

    num_input = dr.num_feature
    num_hidden = 64
    num_output = 1
    max_epoch = 10000
    batch_size = 10
    learning_rate = 0.1
    eps = 1e-6

    params = HyperParameters41(
        learning_rate, max_epoch, batch_size, eps,                        
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier, 
        optimizer_name=OptimizerName.SGD)

    net = Model(dr, num_input, num_hidden, num_output, params)
    ShowResult(net, dr)
