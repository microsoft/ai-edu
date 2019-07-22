# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from HelperClass.NeuralNet_1_0 import *

file_name = "../../data/ch04.npz"

if __name__ == '__main__':
    sdr = DataReader_1_0(file_name)
    sdr.ReadData()
    hp = HyperParameters_1_0(1, 1, eta=0.5, max_epoch=1000, batch_size=-1, eps = 0.02)
    net = NeuralNet_1_0(hp)
    net.train(sdr)
