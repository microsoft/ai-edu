# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.Activators import *


def net(init_method, activator):

    max_epoch = 1
    batch_size = 5
    learning_rate = 0.02
    eps = 0.01

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy3, 
                        init_method,
                        OptimizerName.SGD)

    loss_history = CLossHistory()

    net = NeuralNet(params)

    #num_hidden = [128,112,96,80,64,48,32]
    num_hidden = [128,128,128,128,128,128,128]
    #num_hidden = [100,100,100,100,100,100]
    count = len(num_hidden)-1
    layers = []

    for i in range(count):
        fc = FcLayer(num_hidden[i], num_hidden[i+1], activator)
        layers.append(fc)
        net.add_layer(fc)
    
    # 从正态分布中取1000个样本，每个样本有num_hidden[0]个特征值
    # 转置是为了可以和w1做矩阵乘法
    x = np.random.randn(1000, num_hidden[0]).T
    #x = np.random.normal(size=num_hidden[0]).T

    # 激活函数输出值矩阵列表
    a_value = []
    a_value.append(x)

    # 依次做所有层的前向计算
    for i in range(count):
        a = layers[i].forward(a_value[i])
        a_value.append(a)

    for i in range(count):
        ax = plt.subplot(1, count, i+1)
        ax.set_title("layer" + str(i+1))
        plt.ylim(0,10000)
        if i > 0:
            plt.yticks([])
        ax.hist(a_value[i+1].flatten(), bins=25, range=[0,1])
    #end for
    # super title
    plt.suptitle(init_method.name + " : " + activator.get_name())
    plt.show()

if __name__ == '__main__':
    net(InitialMethod.Normal, Sigmoid())
    net(InitialMethod.Xavier, Sigmoid())
    net(InitialMethod.Xavier, Relu())
    net(InitialMethod.MSRA, Relu())
