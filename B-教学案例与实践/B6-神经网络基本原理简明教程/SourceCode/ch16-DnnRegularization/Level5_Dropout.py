# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from Level0_OverfittingNet_Regression import *

def Model_Dropout(dataReader, num_input, num_hidden, num_output, params):
    net = NeuralNet_4_2(params, "overfitting")

    fc1 = FcLayer_2_0(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    s1 = ActivatorLayer(Sigmoid())
    net.add_layer(s1, "s1")
    
    d1 = DropoutLayer(num_hidden, 0.1)
    net.add_layer(d1, "d1")

    fc2 = FcLayer_2_0(num_hidden, num_hidden, params)
    net.add_layer(fc2, "fc2")
    t2 = ActivatorLayer(Tanh())
    net.add_layer(t2, "t2")

    #d2 = DropoutLayer(num_hidden, 0.2)
    #net.add_layer(d2, "d2")

    fc3 = FcLayer_2_0(num_hidden, num_hidden, params)
    net.add_layer(fc3, "fc3")
    t3 = ActivatorLayer(Tanh())
    net.add_layer(t3, "t3")

    d3 = DropoutLayer(num_hidden, 0.2)
    net.add_layer(d3, "d3")
    
    fc4 = FcLayer_2_0(num_hidden, num_output, params)
    net.add_layer(fc4, "fc4")

    net.train(dataReader, checkpoint=100, need_test=True)
    net.ShowLossHistory(XCoordinate.Epoch)
    
    return net

if __name__ == '__main__':

    dr = LoadData()
    hp, num_hidden = SetParameters()
    net = Model_Dropout(dr, 1, num_hidden, 1, hp)
    ShowResult(net, dr, hp.toString())
