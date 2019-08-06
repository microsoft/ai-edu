# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_1 import *
from MiniFramework.ActivationLayer import *

def net(init_method, activator):

    max_epoch = 1
    batch_size = 5
    learning_rate = 0.02

    params = HyperParameters_4_1(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=init_method)

    net = NeuralNet_4_1(params, "level1")
    num_hidden = [128,128,128,128,128,128,128]
    fc_count = len(num_hidden)-1
    layers = []

    for i in range(fc_count):
        fc = FcLayer_1_1(num_hidden[i], num_hidden[i+1], params)
        net.add_layer(fc, "fc")
        layers.append(fc)

        ac = ActivationLayer(activator)
        net.add_layer(ac, "activator")
        layers.append(ac)
    # end for
    
    # 从正态分布中取1000个样本，每个样本有num_hidden[0]个特征值
    # 转置是为了可以和w1做矩阵乘法
    x = np.random.randn(1000, num_hidden[0])

    # 激活函数输出值矩阵列表
    a_value = []

    # 依次做所有层的前向计算
    input = x
    for i in range(len(layers)):
        output = layers[i].forward(input)
        # 但是只记录激活层的输出
        if isinstance(layers[i], ActivationLayer):
            a_value.append(output)
        # end if
        input = output
    # end for

    for i in range(len(a_value)):
        ax = plt.subplot(1, fc_count+1, i+1)
        ax.set_title("layer" + str(i+1))
        plt.ylim(0,10000)
        if i > 0:
            plt.yticks([])
        ax.hist(a_value[i].flatten(), bins=25, range=[0,1])
    #end for
    # super title
    plt.suptitle(init_method.name + " : " + activator.get_name())
    plt.show()

if __name__ == '__main__':
    net(InitialMethod.Normal, Sigmoid())
    net(InitialMethod.Xavier, Sigmoid())
    net(InitialMethod.Xavier, Relu())
    net(InitialMethod.MSRA, Relu())
