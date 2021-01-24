# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet_4_1 import *
from ExtendedDataReader.MnistImageDataReader import *

file_1 = "../../Data/data_batch_1.bin"
file_2 = "../../Data/data_batch_2.bin"
file_3 = "../../Data/data_batch_3.bin"
file_4 = "../../Data/data_batch_4.bin"
file_5 = "../../Data/data_batch_5.bin"
test_file = "../../Data/test_batch.bin"

def LoadData():
    print("reading data...")
    dr = MnistImageDataReader("vector")
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(NetType.MultipleClassifier)
    dr.GenerateValidationSet(k=20)
    print(dr.num_validation, dr.num_example, dr.num_test, dr.num_train)
    return dr

if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 128
    num_hidden2 = 64
    num_hidden3 = 32
    num_hidden4 = 16
    num_output = 10
    max_epoch = 30
    batch_size = 64
    learning_rate = 0.1

    params = HyperParameters_4_1(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        stopper=Stopper(StopCondition.StopLoss, 0.12))

    net = NeuralNet_4_1(params, "Mnist")

    fc1 = FcLayer_1_1(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    bn1 = BnLayer(num_hidden1)
    net.add_layer(bn1, "bn1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "r1")
    
    fc2 = FcLayer_1_1(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    bn2 = BnLayer(num_hidden2)
    net.add_layer(bn2, "bn2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "r2")

    fc3 = FcLayer_1_1(num_hidden2, num_hidden3, params)
    net.add_layer(fc3, "fc3")
    bn3 = BnLayer(num_hidden3)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "r3")
    
    fc4 = FcLayer_1_1(num_hidden3, num_hidden4, params)
    net.add_layer(fc4, "fc4")
    bn4 = BnLayer(num_hidden4)
    net.add_layer(bn4, "bn4")
    r4 = ActivationLayer(Relu())
    net.add_layer(r4, "r4")

    fc5 = FcLayer_1_1(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ClassificationLayer(Softmax())
    net.add_layer(softmax, "softmax")

    #net.load_parameters()

    net.train(dataReader, checkpoint=0.05, need_test=True)
    
    net.ShowLossHistory(xcoord=XCoordinate.Iteration)
