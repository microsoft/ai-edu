# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass.NeuralNet_1_2 import *

file_name = "../../data/ch09.train.npz"

class DataReaderEx(DataReader_1_3):
    def Add(self):
        X = self.XTrain[:,0:1]**2
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**3
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**4
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**5
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**6
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**7
        self.XTrain = np.hstack((self.XTrain, X))
        X = self.XTrain[:,0:1]**8
        self.XTrain = np.hstack((self.XTrain, X))

def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX1 = np.linspace(0,1,100).reshape(100,1)
    TX2 = np.hstack((TX1, TX1[:,]**2))
    TX3 = np.hstack((TX2, TX1[:,]**3))
    TX4 = np.hstack((TX3, TX1[:,]**4))
    TX5 = np.hstack((TX4, TX1[:,]**5))
    TX6 = np.hstack((TX5, TX1[:,]**6))
    TX7 = np.hstack((TX6, TX1[:,]**7))
    TX8 = np.hstack((TX7, TX1[:,]**8))
    TY = net.inference(TX8)
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
    num_input = 8
    num_output = 1    
    hp = HyperParameters_1_1(num_input, num_output, eta=0.2, max_epoch=50000, batch_size=10, eps=1e-3, net_type=NetType.Fitting)
    #params = HyperParameters(eta=0.2, max_epoch=1000000, batch_size=10, eps=1e-3, net_type=NetType.Fitting)
    net = NeuralNet_1_2(hp)
    net.train(dataReader, checkpoint=500)
    ShowResult(net, dataReader, "Polynomial")
