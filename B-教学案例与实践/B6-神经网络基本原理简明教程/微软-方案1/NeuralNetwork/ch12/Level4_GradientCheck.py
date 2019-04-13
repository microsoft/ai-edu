# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from Level3_ThreeLayerNet import *

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
        dict_params["W" + str(l)] = theta[start:end].reshape((layer_dims[l],layer_dims[l-1]))
        start = end
        end += layer_dims[l]*1
        dict_params["B" + str(l)] = theta[start:end].reshape((layer_dims[l],1))
        start = end
    #end for
    return dict_params

if __name__ == '__main__':
    n_input = 7
    n_hidden1 = 16
    n_hidden2 = 12
    n_output = 10
    eps = 1e-4
    layer_dims = [n_input, n_hidden1, n_hidden2, n_output]
    dict_Param = InitialParameters3(n_input, n_hidden1, n_hidden2, n_output, 2)
    n_example = 1
    x = np.random.randn(n_input, n_example)
    #y = np.array([1,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,0,1]).reshape(-1,n_example)
    #y = np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]).reshape(-1,n_example)
    y = np.array([1,0,0,0,0,0,0,0,0,0]).reshape(-1,n_example)
    
    dict_Cache = forward3(x, dict_Param)
    dict_Grads = backward3(dict_Param, dict_Cache, x, y)
    
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
        J_plus[i] = CalculateLoss(vector_to_dictionary(J_theta_plus, layer_dims), x, y, n_example, forward3)

        J_theta_minus = np.copy(J_theta)
        J_theta_minus[i][0] = J_theta[i][0] - eps
        J_minus[i] = CalculateLoss(vector_to_dictionary(J_theta_minus, layer_dims), x, y, n_example, forward3)

        d_theta_approx[i] = (J_plus[i] - J_minus[i]) / (2 * eps)
    # end for
    numerator = np.linalg.norm(d_theta_real - d_theta_approx)  ####np.linalg.norm 二范数
    denominator = np.linalg.norm(d_theta_approx) + np.linalg.norm(d_theta_real)
    difference = numerator / denominator
    if difference<1e-7:
        print('diference ={} \n NO mistake.'.format(difference))
    else:
        print('diference = {} \n HAS A MISTAKE!!!'.format(difference))
    

