# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from MiniFramework.DataReader_2_0 import *

train_data_name = "../../data/ch17.train_color.npz"
test_data_name = "../../data/ch17.test_color.npz"
name = ["red","green","blue","yellow","cyan","pink"]


def LoadImageData():
    print("reading data...")
    dr = DataReader_2_0(train_data_name, test_data_name)
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(NetType.MultipleClassifier, base=0)
    dr.Shuffle()
    dr.GenerateValidationSet(k=10)
    return dr

def cnn_model():
    num_output = 6
    max_epoch = 50
    batch_size = 16
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "pic_conv")
    
    c1 = ConvLayer((3,28,28), (8,3,3), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    c2 = ConvLayer(p1.output_shape, (16,3,3), (1,0), params)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2") 

    params.learning_rate = 0.1

    f3 = FcLayer_2_0(p2.output_size, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BnLayer(f3.output_size)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")
    
    f4 = FcLayer_2_0(f3.output_size, num_output, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net

def show_samples(x,y,title):
    fig,ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
    for i in range(16):
        ax[i//4,i%4].imshow(x[i].transpose(1,2,0))
        ax[i//4,i%4].set_title(name[np.argmax(y[i])])
        ax[i//4,i%4].axis('off')
    #endfor
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    # for cnn
    dataReader = LoadImageData()
    net = cnn_model()
    x,y = dataReader.GetBatchTrainSamples(16, 0)
    # enf for cnn

    show_samples(x,y,"sample")
    
    net.train(dataReader, checkpoint=0.5, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)
  
    X_test,Y_test = dataReader.GetTestSet()
    for i in range(10):
        start = i * 16
        X = X_test[start:start+16].reshape(16,3,28,28)
        Z = net.inference(X)
        show_samples(X,Z,"predication")
