# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

class CParameters(object):
    def __init__(self, n_example, n_input=1, n_output=1, n_hidden=4, eta=0.1, max_epoch=10000, batch_size=5, lossFunType="MSE", eps=0.001):
        self.num_example = n_example
        self.num_input = n_input
        self.num_output = n_output
        self.num_hidden = n_hidden
        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        if batch_size == -1:
            self.batch_size = self.num_example
        else:
            self.batch_size = batch_size
        # end if
        self.loss_func_type = lossFunType
        self.eps = eps

def InitialParameters(num_input, num_output, flag):
    if flag == 0:
        # zero
        W = np.zeros((num_output, num_input))
    elif flag == 1:
        # normalize
        W = np.random.normal(size=(num_output, num_input))
    elif flag == 2:
        # xavier
        W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                              np.sqrt(6/(num_output+num_input)),
                              size=(num_output,num_input))
    # end if
    B = np.zeros((num_output, 1))
    return W, B


# 获得批样本数据
def GetBatchSamples(X,Y,batch_size,iteration):
    num_feature = X.shape[0]
    start = iteration * batch_size
    end = start + batch_size
    batch_X = X[0:num_feature, start:end].reshape(num_feature, batch_size)
    batch_Y = Y[:, start:end].reshape(-1, batch_size)
    return batch_X, batch_Y


def ReadData(x_data_name, y_data_name):
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XRawData = np.load(Xfile)
        YRawData = np.load(Yfile)
        return XRawData,YRawData
    # end if
    return None,None

