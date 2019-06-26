# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass2.DataReader import *
from HelperClass2.HyperParameters2 import *
from HelperClass2.NeuralNet2 import *
from HelperClass2.Visualizer import *

train_data_name = "../../Data/ch11.train.npz"
test_data_name = "../../Data/ch11.test.npz"

def train(n_hidden):

    n_input = dataReader.num_feature
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.1, 10, 10000
    eps = 0.01

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Bank_2N3")
    net.train(dataReader, 100, True)
    net.ShowTrainingTrace()
    loss = net.GetLatestAverageLoss()

    fig = plt.figure(figsize=(6,6))
    DrawThreeCategoryPoints(dataReader.XTrain[:,0], dataReader.XTrain[:,1], dataReader.YTrain, hp.toString())
    ShowClassificationResult25D(net, 50, str.format("{0}, loss={1:.3f}", hp.toString(), loss))
    plt.show()

if __name__ == '__main__':
    dataReader = DataReader(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeY(YNormalizationMethod.MultipleClassifier, base=1)
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    train(2)
    train(4)
    train(8)
    train(16)
    train(32)
    train(64)
