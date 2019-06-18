# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from HelperClass.NeuralNet import *
from HelperClass.HyperParameters import *

if __name__ == '__main__':
    sdr = SimpleDataReader()
    sdr.ReadData()
    params = HyperParameters(1, 1, eta=0.1, max_epoch=100, batch_size=10, eps = 0.02)
    net = NeuralNet(params)
    net.train(sdr)    
   
