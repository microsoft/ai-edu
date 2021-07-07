# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet_1_2 import *

file_name = "../../data/ch06.npz"
   
# 主程序
if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    # net
    num_input = 2
    num_output = 1
    hp = HyperParameters_1_1(num_input, num_output, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    net = NeuralNet_1_2(hp)
    net.train(reader, checkpoint=1)

    # inference
    x_predicate = np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a = net.inference(x_predicate)
    print("A=", a)




