# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from MiniFramework.NeuralNet_4_2 import *
from Level1_Color_CNN import *

"""
    因为训练数据尺寸太大，不适合于放在github中，所以在运行本程序之前，
    先用SourceCode/Data/ch18_color.py来生成训练数据集
"""

train_data_name = "../../data/ch18.train_color.npz"
test_data_name = "../../data/ch18.test_color.npz"

name = ["red","green","blue","yellow","cyan","pink"]

pos=[19,14,39,4,24,6]

def normalize(x):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)
    return x_n

def visualize_filter_and_layer(net,X):
    fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(8,8))
    N = 6
    C = 2
    # conv1, relu1, pool1
    for i in range(N):
        #z = net.layer_list[0].z
        z = normalize(net.layer_list[0].z)
        for j in range(C):
            ax[0,i].imshow(X[i].transpose(1,2,0))
            ax[0,i].axis('off')

            z[i,j,0,0]=0
            z[i,j,27,27]=1
            ax[j+1,i].imshow(z[i,j])
            ax[j+1,i].axis('off')
    plt.suptitle("color cnn")

    N = 6
    C = 3
    # conv1, relu1, pool1
    for i in range(N):
        z = normalize(net.layer_list[3].z)
        a = normalize(net.layer_list[4].a)
        for j in range(2):
            ax[3,i].imshow(z[i].transpose(1,2,0))
            ax[3,i].axis('off')

            ax[4,i].imshow(a[i].transpose(1,2,0))
            ax[4,i].axis('off')


    plt.show()




if __name__ == '__main__':
    net = cnn_model()
    net.load_parameters()
    dr = LoadData("image")
    X_test, Y_test = dr.GetTestSet()
    X = np.empty((6,3,28,28))
    Y = np.empty(6)
    for i in range(6):
        X[i] = X_test[pos[i]]
        Y[i] = np.argmax(Y_test[pos[i]])
    net.inference(X)

    print(net.layer_list[0].WB.W)    
    print(net.layer_list[0].WB.B)
    visualize_filter_and_layer(net,X)
