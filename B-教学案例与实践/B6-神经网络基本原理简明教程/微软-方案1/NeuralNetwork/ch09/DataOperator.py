# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

class DataOperator(object):
    def __init__(self, method):
        assert(method == "min_max")
        self.method = method

    # normalize data by extracting range from source data
    # return: X_new: normalized data with same shape
    # return: X_norm: 2xN (features)
    #               [[min1, min2, min3...]
    #                [range1, range2, range3...]]
    def NormalizeData(self, X_raw):
        X_new = np.zeros(X_raw.shape)
        num_feature = X_raw.shape[0]
        self.X_norm = np.zeros((2,num_feature))
        # 按行归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            # get one feature from all examples
            x = X_raw[i,:]
            max_value = np.max(x)
            min_value = np.min(x)
            # min value
            self.X_norm[0,i] = min_value 
            # range value
            self.X_norm[1,i] = max_value - min_value 
            x_new = (x - self.X_norm[0,i]) / self.X_norm[1,i]
            X_new[i,:] = x_new
        # end for
        return X_new

    # normalize data by specified range and min_value
    def NormalizePredicateData(self, X_raw):
        X_new = np.zeros(X_raw.shape)
        num_feature = X_raw.shape[0]
        for i in range(num_feature):
            x = X_raw[i,:]
            X_new[i,:] = (x-self.X_norm[0,i])/self.X_norm[1,i]
        return X_new

    @staticmethod
    # read data from file
    def ReadData(x_data_name, y_data_name):
        Xfile = Path(x_data_name)
        Yfile = Path(y_data_name)
        if Xfile.exists() and Yfile.exists():
            XRawData = np.load(Xfile)
            YRawData = np.load(Yfile)
            return XRawData,YRawData
        # end if
        return None,None

    @staticmethod
    # 获得批样本数据
    def GetBatchSamples(X,Y,batch_size,iteration):
        num_feature = X.shape[0]
        start = iteration * batch_size
        end = start + batch_size
        batch_X = X[0:num_feature, start:end].reshape(num_feature, batch_size)
        batch_Y = Y[:, start:end].reshape(-1, batch_size)
        return batch_X, batch_Y

    @staticmethod
    def ToOneHot(YData, num_category):
        num_example = YData.shape[1]
        Y = np.zeros((num_category, num_example))
        for i in range(num_example):
            if YData[0,i] == 1:
                Y[0,i] = 1
            elif YData[0,i] == 2:
                Y[1,i] = 1
            elif YData[0,i] == 3:
                Y[2,i] = 1
            # end if
        # end for
        return Y
