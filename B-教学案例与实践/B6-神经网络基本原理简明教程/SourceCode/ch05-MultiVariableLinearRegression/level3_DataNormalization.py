# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass.NeuralNet_1_1 import *

file_name = "../../data/ch05.npz"

def ShowResult(net, reader):
    # draw example points
    X,Y = reader.GetWholeTrainSamples()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],Y)
    # draw fitting surface
    p = np.linspace(0,1)
    q = np.linspace(0,1)
    P,Q = np.meshgrid(p,q)
    R = np.hstack((P.ravel().reshape(2500,1), Q.ravel().reshape(2500,1)))
    Z = net.inference(R)
    Z = Z.reshape(50,50)
    ax.plot_surface(P,Q,Z, cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    # data
    reader = DataReader_1_1(file_name)
    reader.ReadData()
    reader.NormalizeX()
    # use this setting for the first time
    hp = HyperParameters_1_0(2, 1, eta=0.1, max_epoch=10, batch_size=1, eps = 1e-5)
    # use this setting when you want to train more loops
    #hp = HyperParameters_1_0(2, 1, eta=0.01, max_epoch=500, batch_size=10, eps = 1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
    # inference
    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    print("z=", net.inference(x))

    ShowResult(net, reader)