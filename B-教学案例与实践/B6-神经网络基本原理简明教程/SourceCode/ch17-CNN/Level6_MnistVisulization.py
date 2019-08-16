# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import matplotlib.pyplot as plt

from Level6_MnistConvNet import *

def LoadLessData():
    mdr = MnistImageDataReader("image")
    mdr.ReadLessData(100)
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr

def normalize(x):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)
    return x_n

if __name__ == '__main__':
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
                    plt.imshow(w_n[i,j], cmap='gray')
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
            plt.imshow(z[i,j], cmap='gray')
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