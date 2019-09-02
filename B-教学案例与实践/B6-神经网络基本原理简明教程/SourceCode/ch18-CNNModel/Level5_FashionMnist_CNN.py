# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet_4_2 import *
from ExtendedDataReader.MnistImageDataReader import *

from Level5_FashionMnist_DNN import *

def cnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_cnn")
    
    c1 = ConvLayer((1,28,28), (32,3,3), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 
    """
    c2 = ConvLayer(p1.output_shape, (16,3,3), (1,1), params)
    net.add_layer(c2, "23")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")  
    """
    f3 = FcLayer_2_0(p1.output_size, 128, params)
    net.add_layer(f3, "f3")
    bn3 = BnLayer(f3.output_size)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f4 = FcLayer_2_0(f3.output_size, 10, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net

if __name__ == '__main__':
    mode = "image"
    dataReader = LoadData(mode)
    net = cnn_model()
    net.train(dataReader, checkpoint=0.1, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)

    X_test,Y_test = dataReader.GetTestSet()
    count = 36
    X = X_test[0:count]
    Y = Y_test[0:count]
    Z = net.inference(X)
    show_result(X,np.argmax(Y,axis=1),np.argmax(Z,axis=1),"cnn predication",mode)
