# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass.NeuralNet import *
from HelperClass.Activators import *
from HelperClass.SimpleDataReader import *
from Level3_ShowBinaryResult import *

# step 1
class TanhNeuralNet(NeuralNet):
    def forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        if self.params.net_type == NetType.BinaryClassifier:
            A = Tanh().forward(Z)
            return A
        else:
            return Z
    # setp 1
    """
    def backwardBatch(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        dZ = (batch_a - batch_y) * (1 + batch_a) / batch_a
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB
    """
    # step 2
    def backwardBatch(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        dZ = 2 * (batch_a - batch_y)
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB

class SimpleDataReader_tanh(SimpleDataReader):
    def ToZeroOne(self):
        Y = np.zeros((self.num_train, 1))
        for i in range(self.num_train):
            if self.YTrain[i,0] == 0:     # 第一类的标签设为0
                Y[i,0] = -1
            elif self.YTrain[i,0] == 1:   # 第二类的标签设为1
                Y[i,0] = 1
            # end if
        # end for
        self.YTrain = Y
        self.YRaw = Y
    #end def
#end class

# 主程序
if __name__ == '__main__':
    # data
    reader = SimpleDataReader_tanh()
    reader.ReadData()
    reader.ToZeroOne()
    # net
    params = HyperParameters(eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryTanh)
    num_input = 2
    num_output = 1
    net = TanhNeuralNet(params, num_input, num_output)
    net.train(reader, checkpoint=0.1)

    # show result
    draw_source_data(net, reader)
    draw_predicate_data(net)
    draw_split_line(net)
    plt.show()
