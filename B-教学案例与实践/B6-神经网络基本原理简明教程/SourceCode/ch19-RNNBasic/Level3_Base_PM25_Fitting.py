# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np
from Level3_Base import *

def load_data(net_type, num_step):
    dr = PM25DataReader(net_type, num_step)
    dr.ReadData()
    dr.Normalize()
    dr.GenerateValidationSet(k=1000)
    return dr

def test(net, dataReader, num_step, pred_step, start, end):
    X,Y = dataReader.GetTestSet()
    count = X.shape[0]
    A = np.zeros_like(Y)

    for i in range(0, count, pred_step):
        A[i:i+pred_step] = predict(net, X[i:i+pred_step], num_step, pred_step)

    loss,acc = net.loss_fun.CheckLoss(A, Y)
    print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))

    p1, = plt.plot(A[start:end])
    p2, = plt.plot(Y[start:end])
    plt.legend([p1,p2], ["pred","true"])
    plt.show()

def predict(net, X, num_step, pred_step):
    A = np.zeros((pred_step, 1))
    for i in range(pred_step):
        x = set_predicated_value(X[i:i+1], A, num_step, i)
        a = net.forward(x)
        A[i,0] = a
    #endfor
    return A

def set_predicated_value(X, A, num_step, predicated_step):
    x = X.copy()
    for i in range(predicated_step):
        x[0, num_step - predicated_step + i, 0] = A[i]
    #endfor
    return x


if __name__=='__main__':
    net_type = NetType.Fitting
    output_type = OutputType.LastStep
    num_step = 24
    dataReader = load_data(net_type, num_step)
    eta = 0.1 #0.1
    max_epoch = 50
    batch_size = 64 #64
    num_input = dataReader.num_feature
    num_hidden = 4  # 16
    num_output = dataReader.num_category
    model = str.format("Level3_Fitting_{0}_{1}_{2}_{3}", max_epoch, batch_size, num_hidden, eta)
    hp = HyperParameters_4_3(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden, num_output,
        output_type, net_type)
    n = net(hp, model)
    n.train(dataReader, checkpoint=1)
    pred_steps = [8,4,2,1]
    for i in range(4):
        test(n, dataReader, num_step, pred_steps[i], 1000, 1200)
