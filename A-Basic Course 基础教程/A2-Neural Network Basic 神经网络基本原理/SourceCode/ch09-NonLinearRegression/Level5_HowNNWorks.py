# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass2.NeuralNet_2_0 import *

train_data_name = "../../Data/ch08.train.npz"
test_data_name = "../../Data/ch08.test.npz"

def ShowResult2D(net, title):
    count = 21
    
    TX = np.linspace(0,1,count).reshape(count,1)
    TY = net.inference(TX)

    print("TX=",TX)
    print("Z1=",net.Z1)
    print("A1=",net.A1)
    print("Z=",net.Z2)

    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,np.zeros((count,1)),'.',c='black')
    p2,= plt.plot(TX,net.Z1[:,0],'.',c='r')
    p3,= plt.plot(TX,net.Z1[:,1],'.',c='g')
    plt.legend([p1,p2,p3], ["x","z1","z2"])
    plt.grid()
    plt.show()
    
    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,np.zeros((count,1)),'.',c='black')
    p2,= plt.plot(TX,net.Z1[:,0],'.',c='r')
    p3,= plt.plot(TX,net.A1[:,0],'x',c='r')
    plt.legend([p1,p2,p3], ["x","z1","a1"])
    plt.grid()
    plt.show()

    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,np.zeros((count,1)),'.',c='black')
    p2,= plt.plot(TX,net.Z1[:,1],'.',c='g')
    p3,= plt.plot(TX,net.A1[:,1],'x',c='g')
    plt.legend([p1,p2,p3], ["x","z2","a2"])
    plt.show()

    fig = plt.figure(figsize=(6,6))
    p1,= plt.plot(TX,net.A1[:,0],'.',c='r')
    p2,= plt.plot(TX,net.A1[:,1],'.',c='g')
    p3,= plt.plot(TX,net.Z2[:,0],'x',c='blue')
    plt.legend([p1,p2,p3], ["a1","a2","z"])
    plt.show()

if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    n_input, n_hidden, n_output = 1, 2, 1
    eta, batch_size, max_epoch = 0.05, 10, 5000
    eps = 0.001

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet_2_0(hp, "sin_121")

    net.LoadResult()
    print(net.wb1.W)
    print(net.wb1.B)
    print(net.wb2.W)
    print(net.wb2.B)

    #net.train(dataReader, 50, True)
    #net.ShowTrainingHistory_2_0()
    #ShowResult(net, dataReader, hp.toString())
    ShowResult2D(net, hp.toString())

