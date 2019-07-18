# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level2_XorGateProcess import *

def train(hidden, dataReader):
    n_input = dataReader.num_feature
    n_hidden = hidden
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005
    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet_2_1(hp, "Xor_2N1")
    net.train(dataReader, 100, False)
    epoch = net.GetEpochNumber()
    ShowResultContour(net, dataReader, str.format("{0},epoch={1}", hp.toString(), epoch))

if __name__ == '__main__':
    dataReader = XOR_DataReader()
    dataReader.ReadData()

    train(1, dataReader)
    train(2, dataReader)
    train(3, dataReader)
    train(4, dataReader)
    train(8, dataReader)
    train(16, dataReader)



    
