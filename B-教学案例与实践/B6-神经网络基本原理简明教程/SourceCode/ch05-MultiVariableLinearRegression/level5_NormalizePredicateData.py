# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from HelperClass.NeuralNet11 import *

if __name__ == '__main__':
    # data
    reader = DataReader11()
    reader.ReadData()
    reader.NormalizeX()
    # net
    params = HyperParameters10(2, 1, eta=0.01, max_epoch=100, batch_size=10, eps = 1e-5)
    net = NeuralNet11(params)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    print("Z=", z)

