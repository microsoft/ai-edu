# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *

from MnistImageDataReader import *

train_image_file = '../../Data/train-images-10'
train_label_file = '../../Data/train-labels-10'
test_image_file = '../../Data/test-images-10'
test_label_file = '../../Data/test-labels-10'

def LoadData():
    mdr = MnistImageDataReader(train_image_file, train_label_file, test_image_file, test_label_file, "vector")
    mdr.ReadData()
    mdr.Normalize()
    mdr.GenerateDevSet()
    return mdr

if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 64
    num_hidden2 = 32
    num_hidden3 = 16
    num_output = 10
    max_epoch = 20
    batch_size = 64
    learning_rate = 0.01
    eps = 0.08

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.CrossEntropy3, 
        InitialMethod.MSRA, 
        OptimizerName.Momentum)

    net = NeuralNet(params, "Mnist")

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivatorLayer(Relu())
    net.add_layer(r1, "r1")

    fc2 = FcLayer(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    r2 = ActivatorLayer(Relu())
    net.add_layer(r2, "r2")

    fc3 = FcLayer(num_hidden2, num_hidden3, params)
    net.add_layer(fc3, "fc3")
    r3 = ActivatorLayer(Relu())
    net.add_layer(r3, "r3")

    fc4 = FcLayer(num_hidden3, num_output, params)
    net.add_layer(fc4, "fc4")
    softmax = ActivatorLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=0.5, need_test=True)
    
    net.ShowLossHistory(0, None, 0, 1)
    