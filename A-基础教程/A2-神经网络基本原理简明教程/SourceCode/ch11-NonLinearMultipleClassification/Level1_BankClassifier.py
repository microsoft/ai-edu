# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass2.NeuralNet_2_2 import *
from HelperClass2.Visualizer_1_1 import *

train_data_name = "../../Data/ch11.train.npz"
test_data_name = "../../Data/ch11.test.npz"

if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=1)

    fig = plt.figure(figsize=(6,6))
    DrawThreeCategoryPoints(dataReader.XTrainRaw[:,0], dataReader.XTrainRaw[:,1], dataReader.YTrain, "Source Data")
    plt.show()

    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden = 3
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.1, 10, 5000
    eps = 0.1

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet_2_2(hp, "Bank_233")

    net.LoadResult()


    #net.train(dataReader, 100, True)
    #net.ShowTrainingHistory()

    fig = plt.figure(figsize=(6,6))
    DrawThreeCategoryPoints(dataReader.XTrain[:,0], dataReader.XTrain[:,1], dataReader.YTrain, hp.toString())
    ShowClassificationResult25D(net, 50, hp.toString())
    plt.show()
