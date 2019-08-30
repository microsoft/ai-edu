# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import matplotlib.pyplot as plt

from Level7_ShapeConvNet import *



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

def to3level(w):
    min_v = np.min(w)
    max_v = np.max(w)
    l2 = (max_v + min_v) / 2
    l1 = (l2+min_v) / 2
    l3 = (max_v+l2) / 2
    new_w = np.zeros(w.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i,j] < l2:
                new_w[i,j] = min_v
            else:
                new_w[i,j] = max_v
            """
            if w[i,j] < l1:
                new_w[i,j] = min_v
            elif w[i,j] > l3:
                new_w[i,j] = max_v
            else:
                new_w[i,j] = l2
            """
            #endif
        #endfor
    #endfor
    return new_w

def visulize():
    net = cnn_model()
    net.load_parameters()

    # conv layer 1 kernal
    
    w = net.layer_list[0].WB.W
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12,8))
    for i in range(w.shape[0]):
        #new_w = to3level(w[i,0])
        ax[i//4,i%4].imshow(w[i,0], cmap='gray')
    plt.show()
    
    
    dr = LoadImageData()
    x, y = dr.GetBatchTrainSamples(16, 0)
    net.inference(x)
    N = 16
    C = 8
    
    # conv1, relu1, pool1
    for j in range(N):
        z = normalize(net.layer_list[0].z)
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
        for i in range(C):
            ax[i//4,i%4].imshow(z[j,i])
            ax[i//4,i%4].axis('off')
        plt.show()
    #endfor
    plt.suptitle("conv1-relu1-pool1")



if __name__ == '__main__':
    visulize()
    #deconv()
    