# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet_1_1 import *

file_name = "../../data/ch05.npz"

# get real weights
def DeNormalizeWeightsBias(net, dataReader):
    W_true = np.zeros_like(net.W)
    for i in range(W_true.shape[0]):
        W_true[i,0] = net.W[i,0] / dataReader.X_norm[i,1]
    #end for
    B_true = net.B - W_true[0,0] * dataReader.X_norm[0,0] - W_true[1,0] * dataReader.X_norm[1,0]
    return W_true, B_true

if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX()
    # net
    hp = HyperParameters_1_0(2, 1, eta=0.01, max_epoch=500, batch_size=10, eps = 1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    W_true, B_true = DeNormalizeWeightsBias(net, reader)
    print("W_true=", W_true)
    print("B_true=", B_true)

    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    z = np.dot(x, W_true) + B_true
    print("Z=", z)