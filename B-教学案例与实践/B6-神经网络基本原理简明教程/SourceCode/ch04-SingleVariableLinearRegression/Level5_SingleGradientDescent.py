# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from HelperClass.NeuralNet10 import *

if __name__ == '__main__':
    sdr = DataReader10()
    sdr.ReadData()
    params = HyperParameters10(1, 1, eta=0.1, max_epoch=100, batch_size=1, eps = 0.02)
    net = NeuralNet10(params)
    net.train(sdr)
