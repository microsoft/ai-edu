# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# warning: 运行本程序将会得到失败的结果，这是by design的，是为了讲解课程内容，后面的程序中会有补救的方法

import numpy as np

from HelperClass.NeuralNet_1_1 import *

file_name = "../../data/ch05.npz"

if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    # net
    hp = HyperParameters_1_0(2, 1, eta=0.1, max_epoch=10, batch_size=1, eps = 1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    print(net.inference(x))
