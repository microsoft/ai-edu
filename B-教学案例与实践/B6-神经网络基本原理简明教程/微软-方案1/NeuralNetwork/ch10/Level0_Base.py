# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import struct
import matplotlib.pyplot as plt

'''
train_image_file = 'train-images-01'
train_label_file = 'train-labels-01'
test_image_file = 'test-images-01'
test_label_file = 'test-labels-01'
'''
train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'


# output array: 768 x num_images
def ReadImageFile(image_file_name):
    f = open(image_file_name, "rb")
    a = f.read(4)
    b = f.read(4)
    num_images = int.from_bytes(b, byteorder='big')
    c = f.read(4)
    num_rows = int.from_bytes(c, byteorder='big')
    d = f.read(4)
    num_cols = int.from_bytes(d, byteorder='big')

    image_size = num_rows * num_cols    # 784
    fmt = '>' + str(image_size) + 'B'
    image_data = np.empty((image_size, num_images)) # 784 x M
    for i in range(num_images):
        bin_data = f.read(image_size)
        unpacked_data = struct.unpack(fmt, bin_data)
        array_data = np.array(unpacked_data)
        array_data2 = array_data.reshape((image_size, 1))
        image_data[:,i] = array_data
    f.close()
    return image_data

def ReadLabelFile(lable_file_name, num_output):
    f = open(lable_file_name, "rb")
    f.read(4)
    a = f.read(4)
    num_labels = int.from_bytes(a, byteorder='big')

    fmt = '>B'
    label_data = np.zeros((num_output, num_labels))   # 10 x M
    for i in range(num_labels):
        bin_data = f.read(1)
        unpacked_data = struct.unpack(fmt, bin_data)[0]
        label_data[unpacked_data,i] = 1
    f.close()
    return label_data

def NormalizeData(X):
    X_NEW = np.zeros(X.shape)
    x_max = np.max(X)
    x_min = np.min(X)
    X_NEW = (X - x_min)/(x_max-x_min)
    return X_NEW

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
    A2, dict_Cache = forward(X, dict_Param)
    p = Y * np.log(A2)
    Loss = -np.sum(p) / count
    return Loss

def Test(num_output, dict_Param, num_input, forward):
    raw_data = ReadImageFile(test_image_file)
    X = NormalizeData(raw_data)
    Y = ReadLabelFile(test_label_file, num_output)
    num_images = X.shape[1]
    correct = 0
    for image_idx in range(num_images):
        x = X[:,image_idx].reshape(num_input, 1)
        y = Y[:,image_idx].reshape(num_output, 1)
        A2, dict_Cache = forward(x, dict_Param)
        if np.argmax(A2) == np.argmax(y):
            correct += 1
    return correct, num_images

def Train(X, Y, learning_rate, max_epoch, num_images, num_input, num_output, dict_param, forward, backward, update):
    loss_history = list()

    print("Training...")
    for iteration in range(max_epoch):
        for item in range(num_images):
            x = X[:,item].reshape(num_input,1)
            y = Y[:,item].reshape(num_output,1)
            A2, dict_Cache = forward(x, dict_param)
            dict_Grads = backward(dict_param, dict_Cache, x, y)
            dict_param = update(dict_param, dict_Grads, learning_rate)
            if item % 1000 == 0:
                Loss = CalculateLoss(dict_param, X, Y, num_images, forward)
                print(item, Loss)
                loss_history = np.append(loss_history, Loss)
        print(iteration)

    print("Testing...")
    correct, count = Test(num_output, dict_param, num_input, forward)
    print(str.format("rate={0} / {1} = {2}", correct, count, correct/count))

    plt.plot(loss_history, "r")
    plt.xlabel("Iteration(x1000)")
    plt.ylabel("Loss")
    plt.show()

def LoadData(num_output):
    raw_data = ReadImageFile(train_image_file)
    X = NormalizeData(raw_data)
    Y = ReadLabelFile(train_label_file, num_output)
    return X, Y
