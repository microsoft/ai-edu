# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from HelperClass3.MnistImageDataReader import *
from HelperClass3.HyperParameters3 import *
from HelperClass3.NeuralNet3 import *

if __name__ == '__main__':

    dataReader = MnistImageDataReader(mode="vector")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(YNormalizationMethod.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden1 = 32
    n_hidden2 = 16
    n_output = dataReader.num_category
    eta, eps = 0.2, 0.01
    batch_size, max_epoch = 64, 20

    hp = HyperParameters3(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet3(hp, "MNIST_32_16")
    net.train(dataReader, 0.5, True)
    net.ShowTrainingTrace()
