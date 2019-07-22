# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from HelperClass2.NeuralNet_3_0 import *

# Roll all our parameters dictionary into a single vector satisfying our specific required shape.
def dictionary_to_vector(dict_params):
    keys = []
    count = 0
    for key in ["W1", "B1", "W2", "B2", "W3", "B3"]:
        
        # flatten parameter
        new_vector = np.reshape(dict_params[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]   # -> ["W1","W1",..."b1","b1",..."W2"...]
        
        if count == 0:
            theta = new_vector
        else:         #np.concatenate
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
 
    return theta, keys

# roll all grad values into one vector, the same shape as dictionary_to_vector()
def gradients_to_vector(gradients):
    count = 0
    for key in ["dW1", "dB1", "dW2", "dB2", "dW3", "dB3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
       
        if count == 0:
            d_theta = new_vector
        else:
            d_theta = np.concatenate((d_theta, new_vector), axis=0)
        count = count + 1
 
    return d_theta

# Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
def vector_to_dictionary(theta, layer_dims):
    dict_params = {}
    L = 4  # the number of layers in the networt
    start = 0
    end = 0
    for l in range(1,L):
        end += layer_dims[l]*layer_dims[l-1]
        dict_params["W" + str(l)] = theta[start:end].reshape((layer_dims[l-1],layer_dims[l]))
        start = end
        end += layer_dims[l]*1
        dict_params["B" + str(l)] = theta[start:end].reshape((1,layer_dims[l]))
        start = end
    #end for
    return dict_params

# cross entropy: -Y*lnA
def CalculateLoss(net, dict_Param, X, Y, count, ):
    net.wb1.W = dict_Param["W1"]
    net.wb1.B = dict_Param["B1"]
    net.wb2.W = dict_Param["W2"]
    net.wb2.B = dict_Param["B2"]
    net.wb3.W = dict_Param["W3"]
    net.wb3.B = dict_Param["B3"]
    net.forward(X)
    p = Y * np.log(net.output)
    Loss = -np.sum(p) / count
    return Loss


if __name__ == '__main__':

    n_input = 7
    n_hidden1 = 16
    n_hidden2 = 12
    n_output = 10
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 40

    hp = HyperParameters_3_0(n_input, n_hidden1, n_hidden2, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet_3_0(hp, "MNIST_gradient_check")
    dict_Param = {"W1": net.wb1.W, "B1": net.wb1.B, "W2": net.wb2.W, "B2": net.wb2.B, "W3": net.wb3.W, "B3": net.wb3.B}

    layer_dims = [n_input, n_hidden1, n_hidden2, n_output]
    n_example = 2
    x = np.random.randn(n_example, n_input)
    #y = np.array([1,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,0,1]).reshape(-1,n_example)
    #y = np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]).reshape(-1,n_example)
    y = np.array([1,0,0,0,0,0,0,0,0,0]).reshape(1,-1)
    
    net.forward(x)
    net.backward(x, y)
    dict_Grads = {"dW1": net.wb1.dW, "dB1": net.wb1.dB, "dW2": net.wb2.dW, "dB2": net.wb2.dB, "dW3": net.wb3.dW, "dB3": net.wb3.dB}

    J_theta, keys = dictionary_to_vector(dict_Param)
    d_theta_real = gradients_to_vector(dict_Grads)

    n = J_theta.shape[0]
    J_plus = np.zeros((n,1))
    J_minus = np.zeros((n,1))
    d_theta_approx = np.zeros((n,1))

    # for each of the all parameters in w,b array
    for i in range(n):
        J_theta_plus = np.copy(J_theta)
        J_theta_plus[i][0] = J_theta[i][0] + eps
        # 多分类交叉熵
        J_plus[i] = CalculateLoss(net, vector_to_dictionary(J_theta_plus, layer_dims), x, y, n_example)

        J_theta_minus = np.copy(J_theta)
        J_theta_minus[i][0] = J_theta[i][0] - eps
        J_minus[i] = CalculateLoss(net, vector_to_dictionary(J_theta_minus, layer_dims), x, y, n_example)

        d_theta_approx[i] = (J_plus[i] - J_minus[i]) / (2 * eps)
    # end for
    numerator = np.linalg.norm(d_theta_real - d_theta_approx)  ####np.linalg.norm 二范数
    denominator = np.linalg.norm(d_theta_approx) + np.linalg.norm(d_theta_real)
    difference = numerator / denominator
    print('diference ={}'.format(difference))
    if difference<1e-7:
        print("NO mistake.")
    elif difference<1e-4:
        print("Acceptable, but a little bit high.")
    elif difference<1e-2:
        print("May has a mistake, you need check code!")
    else:
        print("HAS A MISTAKE!!!")
    

