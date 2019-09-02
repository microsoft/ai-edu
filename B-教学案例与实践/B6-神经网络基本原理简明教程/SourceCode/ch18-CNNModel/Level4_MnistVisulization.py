# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import matplotlib.pyplot as plt

from Level4_MnistConvNet import *

def normalize(x):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)
    return x_n

def deconv():
    print("loading data...")
    dataReader = LoadLessData()
    net = model()
    net.load_parameters()
    
    print("forward...")
    # forward
    x, y = dataReader.GetBatchTrainSamples(12, 0)
    print(x.shape)

    data = net.layer_list[0].forward(x)    # conv
    print(data.shape)

    data = net.layer_list[1].forward(data)    # relu
    print(data.shape)

    data = net.layer_list[2].forward(data)    # pooling
    print(data.shape)

    output = data
    output = normalize(data)
    fig,ax = plt.subplots(nrows=3, ncols=3, figsize=(12,12))
    for i in range(8):
        ax[i//3,i%3].imshow(output[7,i])
    plt.show()

    #i = np.argmax(np.sum(output, axis=0))


    for i in range(8):
        output[i,:,:,:]=output[7,:,:,:]

    for i in range(8):
        for j in range(8):
            output[i,j,:,:]=output[i,i,:,:]

    

    print("backward...")
    # backward
    data = net.layer_list[2].backward(output, 1)    # pooling
    print(data.shape)

    data = net.layer_list[1].forward(data, 1)    # relu, using forward as backward
    print(data.shape)

    data = net.layer_list[0].backward(data, 1)    # conv
    print(data.shape)

    output = normalize(data)

    fig,ax = plt.subplots(nrows=2, ncols=12, figsize=(12,5))
    for i in range(12):
        ax[0,i].imshow(x[i,0])
        ax[1,i].imshow(output[i,0])
    plt.show()


def visualize_filter_and_layer_1(net,dataReader):
    # conv layer 1 kernal
    w = net.layer_list[0].WB.W
    fig, ax = plt.subplots(nrows=5, ncols=8, figsize=(12,8))
    for i in range(w.shape[0]):
        ax[0,i].imshow(w[i,0])
        ax[0,i].axis('off')
        ax[1,i].imshow(w[i,0])
        ax[1,i].axis('off')
   
    X, Y = dataReader.GetTestSet()
    net.inference(X[0:20])
    for i in range(20):
        if np.argmax(Y[i]) == 0:
            break

    N = 1
    C = 8
    # conv1, relu1, pool1
    for j in range(3):
        if isinstance(net.layer_list[j], ActivationLayer):
            z = net.layer_list[j].a
        else:
            z = normalize(net.layer_list[j].z)
        for k in range(C):
            ax[j+2,k].imshow(z[i,k])
            ax[j+2,k].axis('off')
    plt.suptitle("layer1:filter,conv,relu,pool")
    plt.show()

if __name__ == '__main__':
    dataReader = LoadData()
    net = model()
    net.load_parameters()
    
    visualize_filter_and_layer_1(net, dataReader)
    #deconv()
    