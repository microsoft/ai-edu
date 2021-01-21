# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass.NeuralNet_1_0 import *

class LogicNotGateDataReader(DataReader_1_0):
    def __init__(self):
        pass

    # x=0,y=1; x=1,y=0
    def ReadData(self):
        X = np.array([0,1]).reshape(2,1)
        Y = np.array([1,0]).reshape(2,1)
        self.XTrain = X
        self.YTrain = Y
        self.num_train = 2

def Test(net):
    z1 = net.inference(0)
    z2 = net.inference(1)
    print (z1,z2)
    if np.abs(z1-1) < 0.001 and np.abs(z2-0)<0.001:
        return True
    return False

def ShowResult(net):
    x = np.array([-0.5,0,1,1.5]).reshape(4,1)
    y = net.inference(x)
    plt.plot(x,y)
    plt.scatter(0,1,marker='^')
    plt.scatter(1,0,marker='o')
    plt.show()

if __name__ == '__main__':
     # read data
    sdr = LogicNotGateDataReader()
    sdr.ReadData()
    # create net
    hp = HyperParameters_1_0(1, 1, eta=0.1, max_epoch=1000, batch_size=1, eps = 1e-8)
    net = NeuralNet_1_0(hp)
    net.train(sdr)
    # result
    print("w=%f,b=%f" %(net.w, net.b))
    # predication
    print(Test(net))
    ShowResult(net)
