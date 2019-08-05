# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *

train_file = "../../Data/ch09.train.npz"
test_file = "../../Data/ch09.test.npz"

def ShowResult(net, dr):
    fig = plt.figure(figsize=(12,5))

    axes = plt.subplot(1,2,1)
    axes.plot(dr.XTest[:,0], dr.YTest[:,0], '.', c='g')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    axes.plot(TX, TY, 'x', c='r')
    axes.set_title("fitting result")

    axes = plt.subplot(1,2,2)
    y_test_real = net.inference(dr.XTest)
    axes.scatter(y_test_real, y_test_real-dr.YTestRaw, marker='o', label='test data')
    axes.set_title("difference")
    plt.show()

def LoadData():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    #dr.NormalizeX()
    #dr.NormalizeY(YNormalizationMethod.Regression)
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

def model():
    dataReader = LoadData()
    num_input = 1
    num_hidden1 = 4
    num_output = 1

    max_epoch = 10000
    batch_size = 10
    learning_rate = 0.5

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.001))

    net = NeuralNet_4_0(params, "Level1_CurveFittingNet")
    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivationLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    fc2 = FcLayer_1_0(num_hidden1, num_output, params)
    net.add_layer(fc2, "fc2")

    net.train(dataReader, checkpoint=100, need_test=True)

    net.ShowLossHistory()
    ShowResult(net, dataReader)

if __name__ == '__main__':
    model()
