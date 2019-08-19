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

cat_pic = "../../Data/cat.png"

if __name__ == '__main__':

    img = cv2.imread(cat_pic)
    plt.imshow(img)
    plt.show()
    print(img.shape)
    data = np.transpose(img, axes=(2,1,0)).reshape((1,3,611,408))
    hp = HyperParameters_4_2(
        0.1, 10, 1,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)
    conv = ConvLayer((3,611,408),(1,5,5),(1,0),hp)
    conv.initialize("know_cnn", "name")
    
    z = conv.forward(data)
    print(z.shape)
    plt.imshow(z[0,0].T)
    plt.show()
