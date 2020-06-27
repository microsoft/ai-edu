# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from ExtendedDataReader.GeometryDataReader import *

train_data_name = "../../data/ch17.train_color.npz"
test_data_name = "../../data/ch17.test_color.npz"

name = ["red","green","blue","yellow","cyan","pink"]

def LoadData(mode):
    print("reading data...")
    dr = GeometryDataReader(train_data_name, test_data_name, mode)
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(NetType.MultipleClassifier, base=0)
    dr.Shuffle()
    dr.GenerateValidationSet(k=10)
    return dr

def dnn_model():
    num_output = 6
    max_epoch = 100
    batch_size = 16
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "color_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net

def show_samples(x,y,title):
    x = x/255
    fig,ax = plt.subplots(nrows=8, ncols=8, figsize=(11,11))
    for i in range(64):
        ax[i//8,i%8].imshow(x[i].transpose(1,2,0))
        ax[i//8,i%8].set_title(name[y[i,0]])
        ax[i//8,i%8].axis('off')
    #endfor
    plt.suptitle(title)
    plt.show()

def train_dnn():
    dataReader = LoadData('vector')
    net = dnn_model()
    return net, dataReader

if __name__ == '__main__':
    net, dataReader = train_dnn()
    x,y = dataReader.XTrainRaw[0:64], dataReader.YTrainRaw[0:64]
    show_samples(x,y,"color:sample")
    net.train(dataReader, checkpoint=0.5, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)
    
    X_test,Y_test = dataReader.GetTestSet()
    Z = net.inference(X_test[0:64])
    X = dataReader.XTestRaw[0:64]
    show_samples(X,np.argmax(Z,axis=1).reshape(64,1),"dnn:predication")
    