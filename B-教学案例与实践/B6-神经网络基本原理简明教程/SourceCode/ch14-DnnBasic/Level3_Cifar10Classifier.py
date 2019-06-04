# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

from ExtendedDataReader.CifarImageReader import *

file_1 = "..\\Data\\data_batch_1.bin"
file_2 = "..\\Data\\data_batch_2.bin"
file_3 = "..\\Data\\data_batch_3.bin"
file_4 = "..\\Data\\data_batch_4.bin"
file_5 = "..\\Data\\data_batch_5.bin"
test_file = "..\\Data\\test_batch.bin"


def LoadData():
    print("reading data...")
    dr = CifarImageReader(file_1, file_2, file_3, file_4, file_5, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(YNormalizationMethod.MultipleClassifier)
    dr.GenerateValidationSet(k=20)
    print(dr.num_validation, dr.num_example, dr.num_test, dr.num_train)
    return dr

if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 128
    num_hidden2 = 64
    num_hidden3 = 32
    num_hidden4 = 16
    num_output = 10
    max_epoch = 50
    batch_size = 32
    learning_rate = 0.01
    eps = 0.08

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.CrossEntropy3, 
        InitialMethod.MSRA, 
        OptimizerName.SGD)

    net = NeuralNet(params, "Cifar10")

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

    fc4 = FcLayer(num_hidden3, num_hidden4, params)
    net.add_layer(fc4, "fc4")
    r4 = ActivatorLayer(Relu())
    net.add_layer(r4, "r4")
    
    fc5 = FcLayer(num_hidden1, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ActivatorLayer(Softmax())
    net.add_layer(softmax, "softmax")

    #net.load_parameters()

    net.train(dataReader, checkpoint=0.5, need_test=True)
    
    net.ShowLossHistory()
    