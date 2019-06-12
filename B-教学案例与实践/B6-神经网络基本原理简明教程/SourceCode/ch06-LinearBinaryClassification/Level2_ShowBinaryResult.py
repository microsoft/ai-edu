# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass.NeuralNet import *
from HelperClass.HyperParameters import *

def ShowResult(net, dataReader):
    fig = plt.figure(figsize=(6.5,6.5))
    X,Y = dataReader.GetWholeTrainSamples()
    for i in range(200):
        if Y[i,0] == 1:
            plt.scatter(X[i,0], X[i,1], marker='x', c='g')
        else:
            plt.scatter(X[i,0], X[i,1], marker='o', c='r')
        #end if
    #end for

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
    
    plt.show()
    

# 主程序
if __name__ == '__main__':
    # data
    reader = SimpleDataReader()
    reader.ReadData()
    # net
    params = HyperParameters(eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    input = 2
    output = 1
    net = NeuralNet(params, input, output)
    net.train(reader, checkpoint=1)

    # inference
    ShowResult(net, reader)




def ShowData(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        # end if
    # end for
    plt.show()

def ShowResult(X,Y,W,B,xt):
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        # end if
    # end for

    b12 = -B[0,0]/W[0,1]
    w12 = -W[0,0]/W[0,1]
    print(w12,b12)
    x = np.linspace(0,1,10)
    y = w12 * x + b12
    plt.plot(x,y)

    for i in range(xt.shape[1]):
        plt.plot(xt[0,i], xt[1,i], '^', c='b')

    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()


