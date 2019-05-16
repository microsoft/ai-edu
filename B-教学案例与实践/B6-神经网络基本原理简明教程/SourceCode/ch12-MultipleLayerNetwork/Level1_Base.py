# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import struct
import matplotlib.pyplot as plt
import time

from MnistDataReader import *

train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def Sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def Softmax(Z):
    shift_z = Z - np.max(Z)
    exp_z = np.exp(shift_z)
    s = np.sum(exp_z, axis=0)
    A = exp_z / s
    return A

# cross entropy: -Y*lnA
def CalculateLoss(dict_Param, X, Y, count, forward):
    dict_Cache = forward(X, dict_Param)
    p = Y * np.log(dict_Cache["Output"])
    Loss = -np.sum(p) / count
    return Loss

def Test(dataReader, num_output, dict_Param, num_input, forward):
    print("Testing...")

    X = dataReader.XTestSet
    Y = dataReader.YTestSet
    count = Y.shape[1]
    num_images = X.shape[1]
    correct = 0
    for image_idx in range(num_images):
        x = X[:,image_idx].reshape(num_input, 1)
        y = Y[:,image_idx].reshape(num_output, 1)
        dict_Cache = forward(x, dict_Param)
        if np.argmax(dict_Cache["Output"]) == np.argmax(y):
            correct += 1

    print(str.format("rate={0} / {1} = {2}", correct, count, correct/count))
    return correct, num_images

def Train(dataReader, learning_rate, max_epoch, num_images, num_input, num_output, dict_param, forward, backward, update, batch_size):

    t0 = time.time()

    loss_history = list()
    print("Training...")
    max_iteration = (int)(dataReader.num_example / batch_size)
    for epoch in range(max_epoch):
        for iteration in range(max_iteration):
            batch_x, batch_y = dataReader.GetBatchSamples(batch_size, iteration)
            dict_Cache = forward(batch_x, dict_param)
            dict_Grads = backward(dict_param, dict_Cache, batch_x, batch_y)
            dict_param = update(dict_param, dict_Grads, learning_rate)
            if iteration % 1000 == 0:
                Loss = CalculateLoss(dict_param, dataReader.X, dataReader.Y, num_images, forward)
                print(epoch, iteration, Loss)
                loss_history = np.append(loss_history, Loss)
            # end if
        # end for
        dataReader.Shuffle()

    t1 = time.time()
    print("time used:", t1 - t0)

    ShowLoss(loss_history)
    return dict_param

def LoadData(num_output):
    mdr = MnistDataReader(train_image_file, train_label_file, test_image_file, test_label_file)
    mdr.ReadData()
    mdr.Normalize()
    return mdr

def ShowLoss(loss_history):
    plt.plot(loss_history, "r")
    plt.xlabel("Iteration(x1000)")
    plt.ylabel("Loss")
    plt.show()
