# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet_1_2 import *

file_name = "../../data/ch07.npz"

def inference(net, reader):
    xt_raw = np.array([5,1,7,6,5,6,2,7]).reshape(4,2)
    xt = reader.NormalizePredicateData(xt_raw)
    output = net.inference(xt)
    r = np.argmax(output, axis=1)+1
    print("output=", output)
    print("r=", r)

# 主程序
if __name__ == '__main__':
    num_category = 3
    reader = DataReader_1_3(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(num_category, base=1)

    num_input = 2
    params = HyperParameters_1_1(num_input, num_category, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet_1_2(params)
    net.train(reader, checkpoint=1)

    inference(net, reader)
