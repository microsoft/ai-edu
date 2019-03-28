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

def NormalizeByRow(X):
    X_NEW = np.zeros(X.shape)
    # get number of features
    n = X.shape[0]
    for i in range(n):
        x_row = X[i,:]
        x_max = np.max(x_row)
        x_min = np.min(x_row)
        if x_max != x_min:
            x_new = (x_row - x_min)/(x_max-x_min)
            X_NEW[i,:] = x_new
    return X_NEW

def NormalizeData(X):
    X_NEW = np.zeros(X.shape)
    x_max = np.max(X)
    x_min = np.min(X)
    X_NEW = (X - x_min)/(x_max-x_min)
    return X_NEW

def Sigmoid(z):
    a=1/(1+np.exp(-z))
    return a

def Tanh(z):
    a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
    return a


def Softmax(Z):
    shift_z = Z - np.max(Z)
    exp_z = np.exp(shift_z)
    s = np.sum(exp_z, axis=0)
    A = exp_z / s
    return A

def ForwardCalculation(X, dict_Param):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]
    W3 = dict_Param["W3"]
    B3 = dict_Param["B3"]
    
    Z1 = np.dot(W1,X) + B1
    A1 = Sigmoid(Z1)

    Z2 = np.dot(W2,A1) + B2
    A2 = Tanh(Z2)

    Z3 = np.dot(W3,A2) + B3
    A3 = Softmax(Z3)
    
    dict_Cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, dict_Cache

def BackPropagation(dict_Param,cache,X,Y):
    W1=dict_Param["W1"]
    W2=dict_Param["W2"]
    W3=dict_Param["W3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    Z1=cache["Z1"]
    Z2=cache["Z2"]
    Z3=cache["Z3"]

    dZ3= A3 - Y
    dW3 = np.dot(dZ3, A2.T)
    dB3 = np.sum(dZ3, axis=1, keepdims=True)

    # dZ2 = W3T * dZ3 * dA3
    dZ2 = np.dot(W3.T, dZ3) * (1-A2*A2) # tanh
    dW2 = np.dot(dZ2, A1.T)
    dB2 = np.sum(dZ2, axis=1, keepdims=True)

    # dZ1 = W2T * dZ2 * dA2
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1-A1)   #sigmoid
    dW1 = np.dot(dZ1, X.T)
    dB1 = np.sum(dZ1, axis=1, keepdims=True)

    dict_Grads = {"dW1": dW1, "dB1": dB1, "dW2": dW2, "dB2": dB2, "dW3": dW3, "dB3": dB3}
    return dict_Grads

def UpdateParam(dict_Param, dict_Grads, learning_rate):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]
    W3 = dict_Param["W3"]
    B3 = dict_Param["B3"]

    dW1 = dict_Grads["dW1"]
    dB1 = dict_Grads["dB1"]
    dW2 = dict_Grads["dW2"]
    dB2 = dict_Grads["dB2"]
    dW3 = dict_Grads["dW3"]
    dB3 = dict_Grads["dB3"]

    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    W3 = W3 - learning_rate * dW3
    B3 = B3 - learning_rate * dB3

    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3}
    return dict_Param

# cross entropy: -Y*lnA
def CalculateLoss(dict_Param, X, Y, count):
    A3, dict_Cache = ForwardCalculation(X, dict_Param)
    p = Y * np.log(A3)
    Loss = -np.sum(p) / count
    return Loss

def InitialParameters(num_input, num_hidden1, num_hidden2, num_output, flag):
    if flag == 0:
        # zero
        W1 = np.zeros((num_hidden1, num_input))
        W2 = np.zeros((num_hidden2, num_hidden1))
        W3 = np.zeros((num_output, num_hidden2))
    elif flag == 1:
        # normalize
        W1 = np.random.normal(size=(num_hidden1, num_input))
        W2 = np.random.normal(size=(num_hidden2, num_hidden1))
        W3 = np.random.normal(size=(num_output, num_hidden2))
    elif flag == 2:
        #
        t1 = np.sqrt(6/(num_input + num_hidden1))
        W1 = np.random.uniform(-t1, t1, size=(num_hidden1, num_input))

        t2 = np.sqrt(6/(num_hidden2 + num_hidden1))
        W2 = np.random.uniform(-t2, t2, size=(num_hidden2, num_hidden1))

        t3 = np.sqrt(6/(num_output + num_hidden2))
        W3 = np.random.uniform(-t3, t3, size=(num_output, num_hidden2))

    B1 = np.zeros((num_hidden1, 1))
    B2 = np.zeros((num_hidden2, 1))
    B3 = np.zeros((num_output, 1))
    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3}
    return dict_Param

def Test(num_output, dict_Param, num_input):
    raw_data = ReadImageFile(test_image_file)
    X = NormalizeData(raw_data)
    Y = ReadLabelFile(test_label_file, num_output)
    num_images = X.shape[1]
    correct = 0
    for image_idx in range(num_images):
        x = X[:,image_idx].reshape(num_input, 1)
        y = Y[:,image_idx].reshape(num_output, 1)
        A2, dict_Cache = ForwardCalculation(x, dict_Param)
        if np.argmax(A2) == np.argmax(y):
            correct += 1
    return correct, num_images

if __name__ == '__main__':

    print("Loading...")
    learning_rate = 0.05
    num_hidden1 = 64
    num_hidden2 = 16
    num_output = 10

    raw_data = ReadImageFile(train_image_file)
    X = NormalizeData(raw_data)
    Y = ReadLabelFile(train_label_file, num_output)

    num_images = X.shape[1]
    num_input = X.shape[0]
    max_iteration = 10

    dict_Param = InitialParameters(num_input, num_hidden1, num_hidden2, num_output, 2)

    loss_history = list()
    eps = 1e-1
    print("Training...")
    for iteration in range(max_iteration):
        for item in range(num_images):
            x = X[:,item].reshape(num_input,1)
            y = Y[:,item].reshape(num_output,1)
            A3, dict_Cache = ForwardCalculation(x, dict_Param)
            dict_Grads = BackPropagation(dict_Param, dict_Cache, x, y)
            dict_Param = UpdateParam(dict_Param, dict_Grads, learning_rate)
            if item % 1000 == 0:
                Loss = CalculateLoss(dict_Param, X, Y, num_images)
                print(item, Loss)
                loss_history = np.append(loss_history, Loss)
                if Loss < eps:
                    break
                #end if
            #end if
        #end for
        if Loss < eps:
            break
        print(iteration)

    print("Testing...")
    correct, count = Test(num_output, dict_Param, num_input)
    print(str.format("rate={0} / {1} = {2}", correct, count, correct/count))

    plt.plot(loss_history, "r")
    plt.title("Xavier Initilization")
    plt.xlabel("Iteration(x1000)")
    plt.ylabel("Loss")
    plt.show()
