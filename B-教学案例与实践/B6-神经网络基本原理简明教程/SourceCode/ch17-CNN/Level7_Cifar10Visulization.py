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

def LoadData():
    print("reading data...")
    mdr = CifarImageDataReader("image")
    mdr.ReadLessData()
    return mdr

if __name__ == '__main__':
    dataReader = LoadData()
    net = model()
    net.load_parameters()

    x, y = dataReader.GetBatchTrainSamples(20, 0)
    output = net.inference(x)
    
    z = normalize(net.layer_list[0].z)
    N,C,H,W = z.shape
    print(z.shape)
    for i in range(N):
        for j in range(C):
            idx = 2*100+(C/2)*10+j+1
            print(idx)
            plt.subplot(idx)
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