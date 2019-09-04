# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from Level1_Color_CNN import *

train_data_name = "../../data/ch17.train_color.npz"
test_data_name = "../../data/ch17.test_color.npz"

name = ["red","green","blue","yellow","cyan","pink"]

pos=[19,14,39,4,24,6]

def normalize(x):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)
    return x_n

def visualize_filter_and_layer_2(net,X):
    fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(8,6))
    N = 6
    C = 3
    # conv1, relu1, pool1
    for i in range(6):
        z = normalize(net.layer_list[3].z)
        a = normalize(net.layer_list[4].a)
        for j in range(2):
            ax[0,i].imshow(X[i].transpose(1,2,0))
            ax[0,i].axis('off')

            ax[1,i].imshow(z[i].transpose(1,2,0))
            ax[1,i].axis('off')

            ax[2,i].imshow(a[i].transpose(1,2,0))
            ax[2,i].axis('off')


    plt.suptitle("layer1:filter,conv,relu,pool")
    plt.show()


def visualize_filter_and_layer_1(net):
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(4,2))
    N = 6
    C = 2
    # conv1, relu1, pool1
    for i in range(6):
        z = net.layer_list[0].z
        for j in range(C):
            ax[j,i].imshow(z[i,j], cmap='gray')
            ax[j,i].axis('off')
    plt.suptitle("layer1:filter,conv,relu,pool")
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
    #visualize_filter_and_layer_1(net)
    visualize_filter_and_layer_2(net,X)
