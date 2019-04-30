# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

# coding: utf-8

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet import *
from MiniFramework.GDOptimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.Activators import *
from MiniFramework.ConvLayer import *
from MiniFramework.PoolingLayer import *

from MnistImageReader import *

train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def LoadData(num_output):
    mdr = MnistImageReader(train_image_file, train_label_file, test_image_file, test_label_file)
    mdr.ReadData()
    mdr.Normalize()
    mdr.Shuffle()
    mdr.GenerateDevSet(12)
    return mdr

def Test(dataReader, model):
    correct = 0
    test_batch = 1000
    max_iteration = dataReader.num_test//test_batch
    for i in range(max_iteration):
        x, y = dataReader.GetBatchTestSamples(test_batch, i)
        model.forward(x)
        correct += CalAccuracy(model.output, None, y)
    #end for
    return correct, dataReader.num_test



def CalAccuracy(a, y_onehot, y_label):
    ra = np.argmax(a, axis=0).reshape(-1,1)
    if y_onehot is None:
        ry = y_label
    elif y_label is None:
        ry = np.argmax(y_onehot, axis=0).reshape(-1,1)
    r = (ra == ry)
    correct = r.sum()
    return correct


def net():
    num_output = 10
    dataReader = LoadData(num_output)

    max_epoch = 1
    batch_size = 50
    eta = 0.01
    eps = 0.01
    params = CParameters(eta, max_epoch, batch_size, eps,
                    LossFunctionName.CrossEntropy3, 
                    InitialMethod.Xavier, 
                    OptimizerName.Adam)

    loss_history = CLossHistory()

    net = NeuralNet(params)

    c1 = ConvLayer((1,28,28), (8,3,3), (1,1), Relu(), params)
    net.add_layer(c1)

    c2 = ConvLayer(c1.output_shape, (8,3,3), (1,1), Relu(), params)
    net.add_layer(c2)

    p1 = PoolingLayer(c2.output_shape, (2,2,), 2, PoolingTypes.MAX)
    net.add_layer(p1)

    c3 = ConvLayer(p1.output_shape, (16,3,3), (1,1), Relu(), params)
    net.add_layer(c3)

    c4 = ConvLayer(c3.output_shape, (16,3,3), (1,1), Relu(), params)
    net.add_layer(c4)

    p2 = PoolingLayer(c4.output_shape, (2,2,), 2, PoolingTypes.MAX)
    net.add_layer(p2)

    f1 = FcLayer(p2.output_size, 32, Relu(), params)
    net.add_layer(f1)

    f2 = FcLayer(f1.output_size, 10, Softmax(), params)
    net.add_layer(f2)

    net.train(dataReader, loss_history)

    loss_history.ShowLossHistory(params)

if __name__ == '__main__':
    net()

