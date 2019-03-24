# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
from enum import Enum

# we assume data format looks like:
# example-> 1, 2, 3, 4
# feature1  x  x  x  x
# feature2  x  x  x  x
#-------------------------
# label     y  y  y  y


class DataNormalization(Enum):
    NormalizeX = 1,
    NormalizeY = 2,
    NormalizePredicate = 3

class DataOperator(object):
    def __init__(self, x_file_name, y_file_name):
        self.x_file_name = x_file_name
        self.y_file_name = y_file_name

    # read data from file
    def ReadData(self):
        Xfile = Path(self.x_file_name)
        Yfile = Path(self.y_file_name)
        if Xfile.exists() and Yfile.exists():
            self.XRawData = np.load(Xfile)
            self.YRawData = np.load(Yfile)
            return self.XRawData, self.YRawData
        # end if
        return None,None

    # normalize data by extracting range from source data
    # return: X_new: normalized data with same shape
    # return: X_norm: 2xN (features)
    #               [[min1, min2, min3...]
    #                [range1, range2, range3...]]

    def NormalizeX(self):
        self.XNormlizedData = np.zeros(self.XRawData.shape)
        num_feature = self.XRawData.shape[0]
        self.X_norm = np.zeros((2,num_feature))
        # 按行归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            # get one feature from all examples
            x = self.XRawData[i,:]
            max_value = np.max(x)
            min_value = np.min(x)
            # min value
            self.X_norm[0,i] = min_value 
            # range value
            self.X_norm[1,i] = max_value - min_value 
            x_new = (x - self.X_norm[0,i]) / self.X_norm[1,i]
            self.XNormlizedData[i,:] = x_new
        # end for
        return self.XNormlizedData

    # normalize data by specified range and min_value
    def NormalizePredicateData(self, X_predicate):
        X_new = np.zeros(X_predicate.shape)
        num_feature = X_predicate.shape[0]
        for i in range(num_feature):
            x = X_predicate[i,:]
            X_new[i,:] = (x-self.X_norm[0,i])/self.X_norm[1,i]
        return X_new


    # 获得批样本数据
    def GetBatchSamples(X,Y,batch_size,iteration):
        num_feature = X.shape[0]
        start = iteration * batch_size
        end = start + batch_size
        batch_X = X[0:num_feature, start:end].reshape(num_feature, batch_size)
        batch_Y = Y[:, start:end].reshape(-1, batch_size)
        return batch_X, batch_Y

    def ToOneHot(self, num_category):
        num_example = self.YRawData.shape[1]
        self.Y = np.zeros((num_category, num_example))
        for i in range(num_example):
            if self.YRawData[0,i] == 1:
                self.Y[0,i] = 1
            elif self.YRawData[0,i] == 2:
                self.Y[1,i] = 1
            elif self.YRawData[0,i] == 3:
                self.Y[2,i] = 1
            # end if
        # end for
        return self.Y

    # if use tanh function, need to set negative_value = -1
    def ToZeroOne(YData, positive_label, negative_label, positiva_value = 1, negative_value = 0):
        num_example = YData.shape[1]
        Y = np.zeros((1, num_example))
        for i in range(num_example):
            if YData[0,i] == negative_label:     # 负类的标签设为0
                Y[0,i] = negative_value
            elif YData[0,i] == positive_label:   # 正类的标签设为1
                Y[0,i] = positiva_value
            # end if
        # end for
        return Y

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self,X,Y):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(X.T)
        np.random.seed(seed)
        YP = np.random.permutation(Y.T)
        return XP.T,YP.T

# unit test
if __name__ == '__main__':
    X = np.array([1,2,3,4,5,6,7,8]).reshape(2,4)
    Y = np.array([7,8,9,0])
    print(X,Y)
    dp = DataOperator()
    X,Y=dp.Shuffle(X,Y)
    print(X,Y)
