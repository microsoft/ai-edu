# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np
from Level3_Base import *
from ExtendedDataReader.PM25DataReader import *

def load_data(net_type, num_step):
    dr = PM25DataReader(net_type, num_step)
    dr.ReadData()
    dr.Normalize()
    dr.GenerateValidationSet(k=1000)
    return dr

def test(net, dataReader, num_step, pred_step, start, end):
    X,Y = dataReader.GetTestSet()
    assert(X.shape[0] == Y.shape[0])
    count = X.shape[0] - X.shape[0] % pred_step
    A = np.zeros((count,1))

    for i in range(0, count, pred_step):
        A[i:i+pred_step] = predict(net, X[i:i+pred_step], num_step, pred_step)

    loss,acc = net.loss_fun.CheckLoss(A, Y[0:count])
    print(str.format("pred_step={0}, loss={1:6f}, acc={2:6f}", pred_step, loss, acc))

    plt.plot(A[start+1:end+1], 'r-x', label="Pred")
    plt.plot(Y[start:end], 'b-o', label="True")
    plt.legend()
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
    num_step = 24 # 24
    dataReader = load_data(net_type, num_step)
    eta = 0.05 # 0.05
    max_epoch = 100 # 100
    batch_size = 64 # 64
    num_input = dataReader.num_feature
    num_hidden = 4  # 4
    num_output = dataReader.num_category
    model = str.format("Level3_Fitting_{0}_{1}_{2}_{3}_{4}_{5}_{6}", max_epoch, batch_size, num_step, num_input, num_hidden, num_output, eta)
    hp = HyperParameters_4_3(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden, num_output,
        output_type, net_type)
    n = net(hp, model)

    n.train(dataReader, checkpoint=1)
    n.loss_trace.ShowLossHistory(hp.toString(), XCoordinate.Iteration)
    #n.load_parameters(ParameterType.Last)
    pred_steps = [8,4,2,1]
    for i in range(4):
        test(n, dataReader, num_step, pred_steps[i], 1050, 1150)

    n.load_parameters(ParameterType.Best)
    pred_steps = [8,4,2,1]
    for i in range(4):
        test(n, dataReader, num_step, pred_steps[i], 1050, 1150)
