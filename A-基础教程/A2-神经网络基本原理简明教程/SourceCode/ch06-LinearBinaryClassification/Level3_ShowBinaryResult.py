# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet_1_2 import *
from HelperClass.HyperParameters_1_1 import *
from HelperClass.Visualizer_1_0 import *

file_name = "../../data/ch06.npz"

def draw_split_line(net):
    b12 = -net.B[0,0]/net.W[1,0]
    w12 = -net.W[0,0]/net.W[1,0]
    print("w12=", w12)
    print("b12=", b12)
    x = np.linspace(0,1,10)
    y = w12 * x + b12
    plt.plot(x,y)
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()

def draw_source_data(dataReader, show=False):
    fig = plt.figure(figsize=(6,6))
    X,Y = dataReader.GetWholeTrainSamples()
    DrawTwoCategoryPoints(X[:,0], X[:,1], Y[:,0], show=show)

def draw_predicate_data(net, threshold=0.5):
    x = np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a = net.inference(x)
    print("A=", a)
    DrawTwoCategoryPoints(x[:,0], x[:,1], a[:,0], show=False, isPredicate=True)
    """
    for i in range(3):
        if a[i,0] > threshold:
            plt.scatter(x[i,0], x[i,1], marker='^', c='r', s=200)
        else:
            plt.scatter(x[i,0], x[i,1], marker='^', c='b', s=200)
    """

# 主程序
if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    draw_source_data(reader, show=True)
    # net
    num_input = 2
    num_output = 1    
    hp = HyperParameters_1_1(num_input, num_output, eta=0.1, max_epoch=1000, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    net = NeuralNet_1_2(hp)
    net.train(reader, checkpoint=10)

    # show result
    draw_source_data(reader, show=False)
    draw_predicate_data(net)
    draw_split_line(net)
    plt.show()
