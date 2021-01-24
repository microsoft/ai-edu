# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.DataReader_2_0 import *

import numpy as np
import csv

train_file = "../../Data/ch14.house.train.npz"
test_file = "../../Data/ch14.house.test.npz"

class HouseDataReader(DataReader_2_0):
    def Drop(self):
        self.XTrain = np.delete(self.XTrain, [0,1,8,9], axis=1)
        self.XTrainRaw = np.delete(self.XTrainRaw, [0,1,8,9], axis=1)
        self.XTest = np.delete(self.XTest, [0,1,8,9], axis=1)
        self.XTestRaw = np.delete(self.XTestRaw, [0,1,8,9], axis=1)
        self.num_feature = self.XTrainRaw.shape[1]


def LoadData():
    dr = HouseDataReader(train_file, test_file)
    dr.ReadData()
    #dr.Drop()
    dr.NormalizeX()
    dr.NormalizeY(NetType.Fitting)
    dr.Shuffle()
    dr.GenerateValidationSet(k=10)
    return dr

def ShowResult(net, dr):
    y_test_result = net.inference(dr.XTest[0:1000,:])
    y_test_real = dr.DeNormalizeY(y_test_result)
    plt.scatter(y_test_real, y_test_real-dr.YTestRaw[0:1000,:], marker='o', label='test data')
#    y_train_result = dr.DeNormalizeY(net.inference(dr.XTrain[0:100,:]))
#    plt.scatter(y_train_result, y_train_result-dr.YTestRaw[0:100,:], marker='s', label='train data')

    plt.show()

def Inference(net, dr):
    output = net.inference(dr.XTest)
    real_output = dr.DeNormalizeY(output)
    with open('house_predict.csv','w', newline='') as csvfile:
        f = csv.writer(csvfile, delimiter=' ')
        f.writerow('price')
        for i in range(real_output.shape[0]):
            f.writerow(real_output[i])


def model(dr):
    num_input = dr.num_feature
    num_hidden1 = 32
    num_hidden2 = 16
    num_hidden3 = 8
    num_hidden4 = 4
    num_output = 1

    max_epoch = 1000
    batch_size = 16
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopDiff, 1e-7))

    net = NeuralNet_4_0(params, "HouseSingle")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")

    fc2 = FcLayer_1_0(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "r2")

    fc3 = FcLayer_1_0(num_hidden2, num_hidden3, params)
    net.add_layer(fc3, "fc3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "r3")

    fc4 = FcLayer_1_0(num_hidden3, num_hidden4, params)
    net.add_layer(fc4, "fc4")
    r4 = ActivationLayer(Relu())
    net.add_layer(r4, "r4")

    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")

    net.train(dr, checkpoint=10, need_test=True)
    return net



if __name__ == '__main__':
    dr = LoadData()
    net = model(dr)
    output = net.inference(dr.XTest)
    real_output = dr.DeNormalizeY(output)
    mse = np.sum((dr.YTestRaw - real_output)**2)/dr.YTest.shape[0]/10000
    print("mse=", mse)
    
    net.ShowLossHistory()
    ShowResult(net, dr)
