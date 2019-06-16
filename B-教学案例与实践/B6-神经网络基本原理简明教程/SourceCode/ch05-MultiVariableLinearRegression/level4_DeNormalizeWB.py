# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from HelperClass.NeuralNet import *

# get real weights
def DeNormalizeWeightsBias(net, dataReader):
    W_real = np.zeros_like(net.W)
    for i in range(W_real.shape[0]):
        W_real[i,0] = net.W[i,0] / dataReader.X_norm[i,1]
    #end for
    B_real = net.B - W_real[0,0] * dataReader.X_norm[0,0] - W_real[1,0] * dataReader.X_norm[1,0]
    return W_real, B_real

if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    reader.NormalizeX()
    # net
    params = HyperParameters(2, 1, eta=0.01, max_epoch=500, batch_size=10, eps = 1e-5)
    net = NeuralNet(params)
    net.train(reader, checkpoint=0.1)
    # inference
    W_real, B_real = DeNormalizeWeightsBias(net, reader)
    print("W_real=", W_real)
    print("B_real=", B_real)

    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    z = np.dot(x, W_real) + B_real
    print("Z=", z)