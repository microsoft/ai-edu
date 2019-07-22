# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_1 import *

train_file = "../../data/ch09.train.npz"
test_file = "../../data/ch09.test.npz"

def ShowResult(net, dr, title):
    # draw test data
    X,Y = dr.GetTestSet()
    plt.plot(X[:,0], Y[:,0], 'o', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, c='r')
    plt.title(title)
    plt.show()
#end def

def model(learning_rate, optimizer):
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    hp = HyperParameters_4_1(
        eta=learning_rate, max_epoch=10000, batch_size=10,
        net_type=NetType.Fitting, 
        init_method=InitialMethod.Xavier,
        optimizer_name=optimizer,
        stopper=Stopper(StopCondition.StopLoss, 0.001))

    n_hidden = 3

    net = NeuralNet_4_1(hp, "level3")

    fc1 = FcLayer_1_1(1,n_hidden,hp)
    net.add_layer(fc1, "fc1")

    s1 = ActivationLayer(Sigmoid())
    net.add_layer(s1, "s1")

    fc2 = FcLayer_1_1(n_hidden,1,hp)
    net.add_layer(fc2, "fc2")

    net.train(dr, checkpoint=50, need_test=True)
    title = str.format("lr:{0},op:{1},epoch:{2},ne:{3}", net.hp.eta, net.hp.optimizer_name.name, net.GetEpochNumber(), n_hidden)
    net.ShowLossHistory()
    ShowResult(net, dr, title)

if __name__ == '__main__':
    model(0.3, OptimizerName.AdaGrad)
    model(0.5, OptimizerName.AdaGrad)
    model(0.7, OptimizerName.AdaGrad)

    model(0.1, OptimizerName.AdaDelta)
    model(0.01, OptimizerName.AdaDelta)

    model(0.1, OptimizerName.RMSProp)
    model(0.01, OptimizerName.RMSProp)
    model(0.005, OptimizerName.RMSProp)

    model(0.1, OptimizerName.Adam)
    model(0.01, OptimizerName.Adam)
    model(0.005, OptimizerName.Adam)
    model(0.001, OptimizerName.Adam)
