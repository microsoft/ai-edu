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


if __name__ == '__main__':
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    hp = HyperParameters_4_1(
        eta=0.1, max_epoch=50, batch_size=10,
        net_type=NetType.Fitting, 
        init_method=InitialMethod.Zero,
        optimizer_name=OptimizerName.SGD,
        stopper=Stopper(StopCondition.StopLoss, 0.001))

    n_hidden = 3

    net = NeuralNet_4_1(hp, "level1_ch09")

    fc1 = FcLayer_1_1(1,n_hidden,hp)
    net.add_layer(fc1, "fc1")

    s1 = ActivationLayer(Sigmoid())
    net.add_layer(s1, "s1")

    fc2 = FcLayer_1_1(n_hidden,1,hp)
    net.add_layer(fc2, "fc2")

    net.train(dr, checkpoint=1, need_test=True)
    title = str.format("lr:{0},op:{1},epoch:{2},ne:{3}", net.hp.eta, net.hp.optimizer_name.name, net.GetEpochNumber(), n_hidden)
    net.ShowLossHistory()
    net.PrintWeightsBiasValue()
