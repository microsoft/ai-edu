# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from MiniFramework.ActivatorLayer import *

train_file = "../../Data/ch16.train.npz"
test_file = "../../Data/ch16.test.npz"


def Model(dataReader, num_input, num_hidden, num_output, params):
    net = NeuralNet_4_2(params, "overfitting")

    fc1 = FcLayer_2_0(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    s1 = ActivatorLayer(Sigmoid())
    net.add_layer(s1, "s1")
    
    fc2 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc2, "fc2")
    t2 = ActivatorLayer(Tanh())
    net.add_layer(t2, "t2")
    
    fc3 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc3, "fc3")
    t3 = ActivatorLayer(Tanh())
    net.add_layer(t3, "t3")
    
    fc4 = FcLayer(num_hidden, num_output, params)
    net.add_layer(fc4, "fc4")

    net.train(dataReader, checkpoint=100, need_test=True)
    net.ShowLossHistory(XCoordinate.Epoch, ymin=0.8)
    
    return net

def ShowResult(net, dr, title):
    TX = np.linspace(dr.XTrain.min(),dr.XTrain.max(),100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, c='red')
    plt.title("fitting result")
    plt.scatter(dr.XTrain, dr.YTrain)
    plt.scatter(dr.XTest, dr.YTest, marker='.', c='g')
    plt.title(title)
    plt.show()

def LoadData():
    dr = DataReader20(train_file, test_file)
    dr.ReadData()
    #dr.NormalizeX()
    #dr.NormalizeY(NetType.Fitting)
    #dr.Shuffle()
    return dr

def SetParameters():
    num_hidden = 16
    max_epoch = 20000
    batch_size = 5
    learning_rate = 0.1
    eps = 1e-6
    
    hp = HyperParameters41(
        learning_rate, max_epoch, batch_size, eps,                        
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier)

    return hp, num_hidden

if __name__ == '__main__':
    dr = LoadData()
    hp, num_hidden = SetParameters()
    net = Model(dr, 1, num_hidden, 1, hp)
    ShowResult(net, dr, hp.toString())
