# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass.NeuralNet import *
from HelperClass.SimpleDataReader import *
from HelperClass.HyperParameters import *

file_name = "../../data/ch08.train.npz"

class DataReaderEx(SimpleDataReader):
    def Add(self):
        X = self.XTrain[:,0:1]**2
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**3
        self.XTrain = np.hstack((self.XTrain, X))


def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX1 = np.linspace(0,1,100).reshape(100,1)
    TX = np.hstack((TX1, TX1[:,]**2))
    TX = np.hstack((TX, TX1[:,]**3))

    TY = net.inference(TX)
    plt.plot(TX1, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

if __name__ == '__main__':
    dataReader = DataReaderEx(file_name)
    dataReader.ReadData()
    dataReader.Add()
    print(dataReader.XTrain.shape)

    # net
    params = HyperParameters(eta=0.1, max_epoch=10000, batch_size=10, eps=1e-3, net_type=NetType.Fitting)
    num_input = 3
    num_output = 1
    net = NeuralNet(params, num_input, num_output)
    net.train(dataReader, checkpoint=1)
    ShowResult(net, dataReader, "Polynomial")
