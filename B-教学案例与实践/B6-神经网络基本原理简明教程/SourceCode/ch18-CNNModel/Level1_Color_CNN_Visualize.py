# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from Level1_Color_CNN import *

train_data_name = "../../data/ch17.train_color.npz"
test_data_name = "../../data/ch17.test_color.npz"

name = ["red","green","blue","yellow","cyan","pink"]


if __name__ == '__main__':
    net = cnn_model()
    print(net.layer_list[0].WB.W)    
    print(net.layer_list[0].WB.B)
