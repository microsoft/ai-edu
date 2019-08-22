# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import matplotlib.pyplot as plt

from Level7_Cifar10ConvNet import *

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def normalize(x):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)
    return x_n

def LoadData(count):
    print("reading data...")
    mdr = CifarImageDataReader("image")
    mdr.ReadLessData(count)
    return mdr


def normalize(x):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)
    return x_n

def deconv():
    print("loading data...")
    count=2
    dataReader = LoadData(count)
    net = model()
    net.load_parameters()
    
    print("forward...")
    # forward
    x, y = dataReader.GetBatchTrainSamples(count*10, 0)
    print(x.shape)




    output = net.inference(x)
    print(output)
    print(np.argmax(output, axis=1))
    exit()

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


def visulize():

    dataReader = LoadData(20)
    net = model()
    net.load_parameters()
    x, y = dataReader.GetBatchTrainSamples(20, 0)

    fig,ax = plt.subplots(nrows=4, ncols=5, figsize=(12,5))
    for i in range(20):
        ax[i//5,i%5].imshow(x[i].transpose(1,2,0))
    plt.show()


    output = net.inference(x)
    print(np.argmax(output, axis=1))
    exit()

    # conv layer 1 kernal
    """
    w = net.layer_list[0].WB.W
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12,8))
    for i in range(w.shape[0]):
        new_w = to3level(w[i,0])
        ax[i//4,i%4].imshow(new_w, cmap='gray')
    plt.show()
    """

    i=0
    N = 1
    C = 6
    fig, ax = plt.subplots(nrows=3, ncols=C, figsize=(12,8))
    # conv1, relu1, pool1
    for j in range(3):
        if isinstance(net.layer_list[j], ActivationLayer):
            z = net.layer_list[j].a
        else:
            z = normalize(net.layer_list[j].z)
        for k in range(C):
            ax[j,k].imshow(z[i,k])
            ax[j,k].axis('off')
    plt.suptitle("conv1-relu1-pool1")

    C = 16
    fig, ax = plt.subplots(nrows=6, ncols=C//2, figsize=(12,8))
    # conv2, relu2, pool2
    for j in range(3):
        if isinstance(net.layer_list[j+3], ActivationLayer):
            z = net.layer_list[j+3].a
        else:
            z = normalize(net.layer_list[j+3].z)
        for k in range(C):
            ax[j*2+k//8,k%8].imshow(z[i,k])
            ax[j*2+k//8,k%8].axis('off')
    plt.suptitle("conv2-relu2-pool2")
    plt.show()
    

if __name__ == '__main__':
    visulize()
    #deconv()
    