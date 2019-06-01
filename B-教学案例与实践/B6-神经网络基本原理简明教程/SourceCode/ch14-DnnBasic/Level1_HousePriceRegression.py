# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

import numpy as np

train_file = "../../Data/PM25_Train.npz"
test_file = "../../Data/PM25_Test.npz"


class HouseDataReader(DataReader):
    def Drop(self):
        self.XTrain = np.delete(self.XTrain, [0,1,8,9], axis=1)
        self.XTrainRaw = np.delete(self.XTrainRaw, [0,1,8,9], axis=1)
        self.XTest = np.delete(self.XTest, [0,1,8,9], axis=1)
        self.XTestRaw = np.delete(self.XTestRaw, [0,1,8,9], axis=1)
        self.num_feature = self.XTrainRaw.shape[1]


def LoadData():
    dr = HouseDataReader(train_file, test_file)
    dr.ReadData()
    #dr.Drop()
    dr.NormalizeX()
    dr.NormalizeY(YNormalizationMethod.Regression)
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

def ShowResult(net, dr):
    y_test_result = net.inference(dr.XTest[0:1000,:])
    y_test_real = dr.DeNormalizeY(y_test_result)
    plt.scatter(y_test_real, y_test_real-dr.YTestRaw[0:1000,:], marker='o', label='test data')
#    y_train_result = dr.DeNormalizeY(net.inference(dr.XTrain[0:100,:]))
#    plt.scatter(y_train_result, y_train_result-dr.YTestRaw[0:100,:], marker='s', label='train data')

    plt.show()

if __name__ == '__main__':
    dr = LoadData()

    num_input = dr.num_feature
    num_hidden1 = 32
    num_hidden2 = 16
    num_hidden3 = 8
    num_output = 1

    max_epoch = 1000
    batch_size = 32
    learning_rate = 0.1
    eps = 0.001

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.MSE, 
        InitialMethod.MSRA, 
        OptimizerName.SGD)

    net = NeuralNet(params, "House")

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivatorLayer(Relu())
    net.add_layer(sigmoid1, "sigmoid1")
    
    fc2 = FcLayer(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    sigmoid2 = ActivatorLayer(Relu())
    net.add_layer(sigmoid2, "sigmoid2")

    fc3 = FcLayer(num_hidden2, num_hidden3, params)
    net.add_layer(fc3, "fc3")
    sigmoid3 = ActivatorLayer(Relu())
    net.add_layer(sigmoid3, "sigmoid3")

    
    fc4 = FcLayer(num_hidden3, num_output, params)
    net.add_layer(fc4, "fc4")

    #ShowResult(net, dr)

    net.load_parameters()

    #ShowResult(net, dr)

    net.train(dr, checkpoint=10, need_test=True)
    net.ShowLossHistory()

    ShowResult(net, dr)
