# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


def InitialParameters(num_input, num_hidden, num_output, flag):
    if flag == 0:
        # zero
        W1 = np.zeros((num_hidden, num_input))
        W2 = np.zeros((num_output, num_hidden))
    elif flag == 1:
        # normalize
        W1 = np.random.normal(size=(num_hidden, num_input))
        W2 = np.random.normal(size=(num_output, num_hidden))
    elif flag == 2:
        #
        W1=np.random.uniform(-np.sqrt(6)/np.sqrt(num_input+num_hidden),np.sqrt(6)/np.sqrt(num_hidden+num_input),size=(num_hidden,num_input))
        W2=np.random.uniform(-np.sqrt(6)/np.sqrt(num_output+num_hidden),np.sqrt(6)/np.sqrt(num_output+num_hidden),size=(num_output,num_hidden))

    B1 = np.zeros((num_hidden, 1))
    B2 = np.zeros((num_output, 1))
    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
    return dict_Param


# 获得批样本数据
def GetBatchSamples(X,Y,batch_size,iteration):
    num_feature = X.shape[0]
    start = iteration * batch_size
    end = start + batch_size
    batch_X = X[0:num_feature, start:end].reshape(num_feature, batch_size)
    batch_Y = Y[:, start:end].reshape(-1, batch_size)
    return batch_X, batch_Y
