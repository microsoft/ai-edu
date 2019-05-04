# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.Activators import *

from MnistDataReader import *

train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def LoadData(num_output):
    mdr = MnistDataReader(train_image_file, train_label_file, test_image_file, test_label_file)
    mdr.ReadData()
    mdr.Normalize()
    return mdr


if __name__ == '__main__':

    num_output = 10
    dataReader = LoadData(num_output)
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 128
    num_hidden2 = 96
    num_hidden3 = 64
    num_hidden4 = 32
    max_epoch = 10
    batch_size = 5
    learning_rate = 0.02
    eps = 0.01

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy3, 
                        InitialMethod.Xavier, 
                        OptimizerName.SGD)

    loss_history = CLossHistory()

    net = NeuralNet(params)
    fc1 = FcLayer(num_input, num_hidden1, Relu())
    net.add_layer(fc1, "fc1")
    fc2 = FcLayer(num_hidden1, num_hidden2, Relu())
    net.add_layer(fc2, "fc2")
    fc3 = FcLayer(num_hidden2, num_hidden3, Relu())
    net.add_layer(fc3, "fc3")
    fc4 = FcLayer(num_hidden3, num_hidden4, Relu())
    net.add_layer(fc4, "fc4")
    fc5 = FcLayer(num_hidden4, num_output, Softmax())
    net.add_layer(fc5, "fc5")
    net.train(dataReader, loss_history)
    
    loss_history.ShowLossHistory(params, 0, None, 0, 1)
    
    net.load_parameters()
    print("Testing...")
    correct, count = net.Test(dataReader)
    print(str.format("rate={0} / {1} = {2}", correct, count, correct/count))
