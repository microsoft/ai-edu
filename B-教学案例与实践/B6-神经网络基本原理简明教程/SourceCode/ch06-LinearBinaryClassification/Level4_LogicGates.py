# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass.NeuralNet import *


class LogicDataReader(SimpleDataReader):
    def Read_Logic_AND_Data(self):
        X = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y = np.array([0,0,0,1]).reshape(4,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NAND_Data(self):
        X = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y = np.array([1,1,1,0]).reshape(4,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_OR_Data(self):
        X = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y = np.array([0,1,1,1]).reshape(4,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

    def Read_Logic_NOR_Data(self):
        X = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y = np.array([1,0,0,0]).reshape(4,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

   
def Test(net, reader):
    X,Y = reader.GetWholeTrainSamples()
    A = net.inference(X)
    print(A)
    diff = np.abs(A-Y)
    result = np.where(diff < 1e-2, True, False)
    if result.sum() == 4:
        return True
    else:
        return False

def draw_split_line(net, reader, title):
    w = -net.W[0,0] / net.W[1,0]
    b = -net.B[0,0] / net.W[1,0]
    x = np.array([-0.1,1.1])
    y = w * x + b
    plt.plot(x,y)
   
def draw_source_data(reader, title):
    fig = plt.figure(figsize=(5,5))
    plt.grid()
    X,Y = reader.GetWholeTrainSamples()
    for i in range(reader.num_train):
        if Y[i,0] == 0:
            plt.scatter(X[i,0],X[i,1],marker="o",c='b',s=64)
        else:
            plt.scatter(X[i,0],X[i,1],marker="^",c='r',s=64)
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.title(title)
    

def train(reader, title):
    draw_source_data(reader, title)
    plt.show()
    # net train
    num_input = 2
    num_output = 1
    params = HyperParameters(num_input, num_output, eta=0.5, max_epoch=10000, batch_size=1, eps=2e-3, net_type=NetType.BinaryClassifier)
    net = NeuralNet(params)
    net.train(reader, checkpoint=1)
    # test
    print(Test(net, reader))
    # visualize
    draw_source_data(reader, title)
    draw_split_line(net, reader, title)
    plt.show()

if __name__ == '__main__':
    reader = LogicDataReader()
    reader.Read_Logic_AND_Data()
    train(reader, "Logic AND operator")

    reader = LogicDataReader()
    reader.Read_Logic_NAND_Data()
    train(reader, "Logic NAND operator")

    reader = LogicDataReader()
    reader.Read_Logic_OR_Data()
    train(reader, "Logic OR operator")

    reader = LogicDataReader()
    reader.Read_Logic_NOR_Data()
    train(reader, "Logic NOR operator")

