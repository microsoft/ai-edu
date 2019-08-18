# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import matplotlib.pyplot as plt

from Level6_MnistConvNet import *

def LoadLessData():
    mdr = MnistImageDataReader("image")
    mdr.ReadLessData(100)
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    #mdr.Shuffle()
    #mdr.GenerateValidationSet(k=12)
    return mdr

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

def visulize():
    dataReader = LoadLessData()
    net = model()
    net.load_parameters()
    first_conv_layer = None
    for layer in net.layer_list:
        if isinstance(layer, ConvLayer):
            first_conv_layer = layer
            w = layer.WB.W
            # normalization to 0~1
            w_n = normalize(w)
            N,C,H,W = w.shape
            for i in range(N):
                for j in range(C):
                    idx = 2*100+N*C/2*10+i+1
                    print(idx)
                    plt.subplot(idx)
                    #plt.imshow(w_n[i,j], cmap='gray')
                    plt.imshow(w_n[i,j])
            #endfor
            plt.show()
        #endif
        break
    
    x, y = dataReader.GetBatchTrainSamples(20, 0)
    output = net.inference(x)
    
    z = normalize(first_conv_layer.z)
    N,C,H,W = z.shape
    print(z.shape)
    for i in range(N):
        for j in range(C):
            idx = 2*100+(C/2)*10+j+1
            print(idx)
            plt.subplot(idx)
            #plt.imshow(z[i,j], cmap='gray')
            plt.imshow(z[i,j])
        plt.show()
    
    """
    z = normalize(net.layer_list[3].z)
    N,C,H,W = z.shape
    print(z.shape)
    for i in range(N):
        for j in range(C):
            idx = 2*100+(C/2)*10+j+1
            print(idx)
            plt.subplot(idx)
            plt.imshow(z[i,j], cmap='gray')
        plt.show()
    """

if __name__ == '__main__':
    visulize()
    #deconv()
