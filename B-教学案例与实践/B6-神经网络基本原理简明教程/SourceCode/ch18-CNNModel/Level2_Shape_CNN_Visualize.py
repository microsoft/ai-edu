# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from Level2_Shape_CNN import *

name = ["rect","tri","circle","diamond","line"]
pos=[0,1,3,6,12]

def visualize_filter_and_layer_1(net,dataReader):
    X_test, Y_test = dataReader.GetTestSet()
    X = np.empty((5,1,28,28))
    Y = np.empty(5)
    for i in range(5):
        X[i] = X_test[pos[i]]
        Y[i] = np.argmax(Y_test[pos[i]])
    net.inference(X)
    fig, ax = plt.subplots(nrows=6, ncols=8, figsize=(9,9))
    N = 8
    C = 5

    w = net.layer_list[0].WB.W
    for i in range(w.shape[0]):
        ax[0,i].imshow(w[i,0])
        ax[0,i].axis('off')


    # conv1, relu1, pool1
    for i in range(N):
        z = net.layer_list[0].z
        for j in range(C):
            ax[j+1,i].imshow(z[j,i])
            ax[j+1,i].axis('off')
    plt.suptitle("layer1:filter,conv,relu,pool")
    plt.show()

if __name__ == '__main__':
    net = cnn_model()
    net.load_parameters()
    dr = LoadData("image")
    visualize_filter_and_layer_1(net, dr)
