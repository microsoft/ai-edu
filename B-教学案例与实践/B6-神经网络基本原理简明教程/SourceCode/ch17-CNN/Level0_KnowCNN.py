# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np
import cv2

from MiniFramework.ConvWeightsBias import *
from MiniFramework.ConvLayer import *
from MiniFramework.ActivationLayer import *
from MiniFramework.PoolingLayer import *
from MiniFramework.HyperParameters_4_2 import *
from MiniFramework.jit_utility import *

cat_pic = "../../Data/cat.png"
car_pic = "../../Data/car.png"

def try_filters(file_name):
    img = cv2.imread(file_name)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    batch_size = 1
    input_channel = 1
    (height, width) = img2.shape
    FH = 3
    FW = 3
    print(img2.shape)
    data = np.transpose(img2, axes=(1,0)).reshape((1,1,width,height))
    hp = HyperParameters_4_2(
        0.1, 10, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)
    conv = ConvLayer((1,width,height),(1,FH,FW),(1,1),hp)
    conv.initialize("know_cnn", "name")
    
    filters = [
        np.array([-1,-1,-1,
                  -1,9,-1,
                  -1,-1,-1]),       # sharpness filter
        np.array([0,-1,0,
                  0,2,0,
                  0,-1,0]),         # vertical edge
        np.array([1,1,1,
                  1,-9,1,
                  1,1,1]),          # surround
        np.array([-1,0,1,
                  -2,0,2,
                  -1,0,1]),         # sobel x
        np.array([0,0,0,
                  0,1,0,
                  0,0,0]),          # nothing
        np.array([0,0,0,
                  -1,2,-1,
                  0,0,0]),          # horizontal edge
        np.array([0.1,0.1,0.1,
                  0.1,0.1,0.1,
                  0.1,0.1,0.1]),    # blur
        np.array([-1,-2,-1,
                  0,0,0,
                  1,2,1]),          # sobel y
        np.array([2,0,0,
                  0,-1,0,
                  0,0,-1])]         # embossing

    filters_name = ["sharpness", "vertical edge", "surround", "sobel x edge", "nothing", "horizontal edge", "blur", "sobel y edge", "embossing"]

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9,9))
    z = []
    for i in range(len(filters)):
        filter = np.repeat(filters[i], input_channel).reshape(batch_size, input_channel,FH,FW)
        conv.set_filter(filter, None)
        z.append(conv.forward(data))
        ax[i//3, i%3].imshow(z[i][0,0].T)
        ax[i//3, i%3].set_title(filters_name[i])
    plt.suptitle("try filters")
    plt.show()
    return z

def conv_relu_pool():
    img = cv2.imread(cat_pic)
    #img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    batch_size = 1
    (height, width, input_channel) = img.shape
    FH = 3
    FW = 3
    data = np.transpose(img, axes=(2,1,0)).reshape((batch_size,input_channel,width,height))
    hp = HyperParameters_4_2(
        0.1, 10, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)
    conv = ConvLayer((input_channel,width,height),(1,FH,FW),(1,0),hp)
    conv.initialize("know_cnn", "conv")
    kernal = np.array([ -1,0,1,
                        -2,0,2,
                        -1,0,1])
    filter = np.repeat(kernal, input_channel).reshape(batch_size, input_channel,FH,FW)
    conv.set_filter(filter, None)
    z1 = conv.forward(data)
    z2 = Relu().forward(z1)
    pool = PoolingLayer(z2[0].shape, (2,2), 2, PoolingTypes.MAX)
    pool.initialize("know_cnn", "pool")
    z3 = pool.forward(z2)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    ax[0,0].imshow(img)
    ax[0,0].set_title("source:" + str(img.shape))
    ax[0,1].imshow(z1[0,0].T)
    ax[0,1].set_title("conv:" + str(z1.shape))
    ax[1,0].imshow(z2[0,0].T)
    ax[1,0].set_title("relu:" + str(z2.shape))
    ax[1,1].imshow(z3[0,0].T)
    ax[1,1].set_title("pooling:" + str(z3.shape))

    plt.suptitle("conv-relu-pool")
    plt.show()

    
if __name__ == '__main__':
    try_filters(cat_pic)
    try_filters(car_pic)
    conv_relu_pool()
