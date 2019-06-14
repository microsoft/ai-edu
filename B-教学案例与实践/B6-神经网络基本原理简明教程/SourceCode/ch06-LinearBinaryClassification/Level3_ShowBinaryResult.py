# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet import *
from HelperClass.HyperParameters import *

def draw_split_line(net,):
    b12 = -net.B[0,0]/net.W[1,0]
    w12 = -net.W[0,0]/net.W[1,0]
    print("w12=", w12)
    print("b12=", b12)
    x = np.linspace(0,1,10)
    y = w12 * x + b12
    plt.plot(x,y)
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()

def draw_source_data(net, dataReader):
    fig = plt.figure(figsize=(6.5,6.5))
    X,Y = dataReader.GetWholeTrainSamples()
    for i in range(200):
        if Y[i,0] == 1:
            plt.scatter(X[i,0], X[i,1], marker='x', c='g')
        else:
            plt.scatter(X[i,0], X[i,1], marker='o', c='r')
        #end if
    #end for

def draw_predicate_data(net):
    x = np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a = net.inference(x)
    print("A=", a)
    for i in range(3):
        if a[i,0] > 0.5:
            plt.scatter(x[i,0], x[i,1], marker='^', c='g', s=100)
        else:
            plt.scatter(x[i,0], x[i,1], marker='^', c='r', s=100)
        #end if
    #end for
 

# 主程序
if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    # net
    params = HyperParameters(eta=0.1, max_epoch=10000, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    num_input = 2
    num_output = 1
    net = NeuralNet(params, num_input, num_output)
    net.train(reader, checkpoint=1)

    # show result
    draw_source_data(net, reader)
    draw_predicate_data(net)
    draw_split_line(net)
    plt.show()
