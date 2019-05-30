# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

# x1=0,x2=0,y=0
# x1=0,x2=1,y=1
# x1=1,x2=0,y=1
# x1=1,x2=1,y=0
class XOR_DataReader(DataReader):
    def __init__(self):
        pass

    def ReadData(self):
        self.XTrain = np.array([[0,0],[1,1],[0,1],[1,0]])
        self.YTrain = np.array([0,0,1,1]).reshape(4,1)
        self.num_train = 4
        self.num_feature = 2
        self.num_category = 1
        
        self.XVld = self.XTrain
        self.YVld = self.YTrain

        self.XTest = self.XTrain
        self.YTest = self.YTrain

def ShowResult2D(net):
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(1,2)
            output = net.inference(x)
            if output[0,0] >= 0.5:
                plt.plot(x[0,0], x[0,1], 's', c='m')
            else:
                plt.plot(x[0,0], x[0,1], 's', c='y')
            # end if
        # end for
    # end for
    plt.show()
#end def

if __name__ == '__main__':
    dr = XOR_DataReader()
    dr.ReadData()
    
    num_input = 2
    num_hidden = 2
    num_output = 1

    max_epoch = 10000
    batch_size = 1
    learning_rate = 0.1
    eps = 0.001

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.CrossEntropy2,
        InitialMethod.Xavier, 
        OptimizerName.SGD)

    net = NeuralNet(params, "XOR")

    fc1 = FcLayer(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivatorLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    
    fc2 = FcLayer(num_hidden, num_output, params)
    net.add_layer(fc2, "fc2")
    sigmoid2 = ClassificationLayer(Sigmoid())
    net.add_layer(sigmoid2, "sigmoid2")

    #net.load_parameters()

    net.train(dr, checkpoint=100, need_test=True)
    net.ShowLossHistory()
    ShowResult2D(net)
