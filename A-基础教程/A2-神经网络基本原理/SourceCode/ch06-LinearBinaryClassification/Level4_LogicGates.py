# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass.NeuralNet_1_2 import *
from HelperClass.Visualizer_1_0 import *

class LogicDataReader(DataReader_1_1):
    def __init__(self):
        pass

    def Read_Logic_NOT_Data(self):
        X = np.array([0,1]).reshape(2,1)
        Y = np.array([1,0]).reshape(2,1)
        self.XTrain = self.XRaw = X
        self.YTrain = self.YRaw = Y
        self.num_train = self.XRaw.shape[0]

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

def draw_split_line(net):
    if (net.W.shape[0] == 2):
        w = -net.W[0,0] / net.W[1,0]
        b = -net.B[0,0] / net.W[1,0]
    else:
        w = net.W[0]
        b = net.B[0]
    x = np.array([-0.1,1.1])
    y = w * x + b
    plt.plot(x,y)
   
def draw_source_data(reader, title, show=False):
    fig = plt.figure(figsize=(5,5))
    plt.grid()
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.title(title)
    X,Y = reader.GetWholeTrainSamples()
    if title == "Logic NOT operator":
        DrawTwoCategoryPoints(X[:,0], np.zeros_like(X[:,0]), Y[:,0], title=title, show=show)
    else:
        DrawTwoCategoryPoints(X[:,0], X[:,1], Y[:,0], title=title, show=show)

def train(reader, title):
    draw_source_data(reader, title, show=True)
    # net train
    num_input = reader.XTrain.shape[1]
    num_output = 1
    hp = HyperParameters_1_1(num_input, num_output, eta=0.5, max_epoch=10000, batch_size=1, eps=2e-3, net_type=NetType.BinaryClassifier)
    net = NeuralNet_1_2(hp)
    net.train(reader, checkpoint=1)
    # test
    print(Test(net, reader))
    # visualize
    draw_source_data(reader, title, show=False)
    draw_split_line(net)
    plt.show()

if __name__ == '__main__':
    reader = LogicDataReader()
    reader.Read_Logic_NOT_Data()
    train(reader, "Logic NOT operator")

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

