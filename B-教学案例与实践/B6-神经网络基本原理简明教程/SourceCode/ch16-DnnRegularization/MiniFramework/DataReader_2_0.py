# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 2.0
"""

import numpy as np
from pathlib import Path

from MiniFramework.EnumDef_5_0 import *

"""
X:
    x1: feature1 feature2 feature3...
    x2: feature1 feature2 feature3...
    x3: feature1 feature2 feature3...
    ......

Y:  [if regression, value]
    [if binary classification, 0/1]
    [if multiple classification, e.g. 4 category, one-hot]
"""

class DataReader_2_0(object):
    def __init__(self, train_file, test_file):
        self.train_file_name = train_file
        self.test_file_name = test_file
        self.num_train = 0        # num of training examples
        self.num_test = 0         # num of test examples
        self.num_validation = 0   # num of validation examples
        self.num_feature = 0      # num of features
        self.num_category = 0     # num of categories
        self.XTrain = None        # training feature set
        self.YTrain = None        # training label set
        self.XTest = None         # test feature set
        self.YTest = None         # test label set
        self.XTrainRaw = None     # training feature set before normalization
        self.YTrainRaw = None     # training label set before normalization
        self.XTestRaw = None      # test feature set before normalization
        self.YTestRaw = None      # test label set before normalization
        self.XDev = None          # validation feature set
        self.YDev = None          # validation lable set

    # read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XTrainRaw = data["data"]
            self.YTrainRaw = data["label"]
            assert(self.XTrainRaw.shape[0] == self.YTrainRaw.shape[0])
            self.num_train = self.XTrainRaw.shape[0]
            self.num_feature = self.XTrainRaw.shape[1]
            self.num_category = len(np.unique(self.YTrainRaw))
            # this is for if no normalize requirment
            self.XTrain = self.XTrainRaw
            self.YTrain = self.YTrainRaw
        else:
            raise Exception("Cannot find train file!!!")
        #end if

        test_file = Path(self.test_file_name)
        if test_file.exists():
            data = np.load(self.test_file_name)
            self.XTestRaw = data["data"]
            self.YTestRaw = data["label"]
            assert(self.XTestRaw.shape[0] == self.YTestRaw.shape[0])
            self.num_test = self.XTestRaw.shape[0]
            # this is for if no normalize requirment
            self.XTest = self.XTestRaw
            self.YTest = self.YTestRaw
            # in case there has no validation set created
            self.XDev = self.XTest
            self.YDev = self.YTest
        else:
            raise Exception("Cannot find test file!!!")
        #end if

    # merge train/test data first, normalize, then split again
    def NormalizeX(self):
        x_merge = np.vstack((self.XTrainRaw, self.XTestRaw))
        x_merge_norm = self.__NormalizeX(x_merge)
        train_count = self.XTrainRaw.shape[0]
        self.XTrain = x_merge_norm[0:train_count,:]
        self.XTest = x_merge_norm[train_count:,:]

    def __NormalizeX(self, raw_data):
        temp_X = np.zeros_like(raw_data)
        self.X_norm = np.zeros((2, self.num_feature))
        # 按行归一化,即所有样本的同一特征值分别做归一化
        for i in range(self.num_feature):
            # get one feature from all examples
            x = raw_data[:, i]
            max_value = np.max(x)
            min_value = np.min(x)
            # min value
            self.X_norm[0,i] = min_value 
            # range value
            self.X_norm[1,i] = max_value - min_value 
            x_new = (x - self.X_norm[0,i]) / self.X_norm[1,i]
            temp_X[:, i] = x_new
        # end for
        return temp_X

    def NormalizeY(self, nettype, base=0):
        if nettype == NetType.Fitting:
            y_merge = np.vstack((self.YTrainRaw, self.YTestRaw))
            y_merge_norm = self.__NormalizeY(y_merge)
            train_count = self.YTrainRaw.shape[0]
            self.YTrain = y_merge_norm[0:train_count,:]
            self.YTest = y_merge_norm[train_count:,:]                
        elif nettype == NetType.BinaryClassifier:
            self.YTrain = self.__ToZeroOne(self.YTrainRaw, base)
            self.YTest = self.__ToZeroOne(self.YTestRaw, base)
        elif nettype == NetType.MultipleClassifier:
            self.YTrain = self.__ToOneHot(self.YTrainRaw, base)
            self.YTest = self.__ToOneHot(self.YTestRaw, base)

    def __NormalizeY(self, raw_data):
        assert(raw_data.shape[1] == 1)
        self.Y_norm = np.zeros((2,1))
        max_value = np.max(raw_data)
        min_value = np.min(raw_data)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[1, 0] = max_value - min_value 
        y_new = (raw_data - min_value) / self.Y_norm[1, 0]
        return y_new

    def DeNormalizeY(self, predict_data):
        real_value = predict_data * self.Y_norm[1,0] + self.Y_norm[0,0]
        return real_value

    def __ToOneHot(self, Y, base=0):
        count = Y.shape[0]
        temp_Y = np.zeros((count, self.num_category))
        for i in range(count):
            n = (int)(Y[i,0])
            temp_Y[i,n-base] = 1
        return temp_Y

    # for binary classifier
    # if use tanh function, need to set negative_value = -1
    def __ToZeroOne(Y, positive_label=1, negative_label=0, positiva_value=1, negative_value=0):
        temp_Y = np.zeros_like(Y)
        for i in range():
            if Y[i,0] == negative_label:     # 负类的标签设为0
                temp_Y[i,0] = negative_value
            elif Y[i,0] == positive_label:   # 正类的标签设为1
                temp_Y[i,0] = positiva_value
            # end if
        # end for
        return temp_Y

    # normalize data by specified range and min_value
    def NormalizePredicateData(self, X_predicate):
        X_new = np.zeros(X_predicate.shape)
        n_feature = X_predicate.shape[0]
        for i in range(n_feature):
            x = X_predicate[i,:]
            X_new[i,:] = (x-self.X_norm[0,i])/self.X_norm[1,i]
        return X_new

    # need explicitly call this function to generate validation set
    def GenerateValidationSet(self, k = 10):
        self.num_validation = (int)(self.num_train / k)
        self.num_train = self.num_train - self.num_validation
        # validation set
        self.XDev = self.XTrain[0:self.num_validation]
        self.YDev = self.YTrain[0:self.num_validation]
        # train set
        self.XTrain = self.XTrain[self.num_validation:]
        self.YTrain = self.YTrain[self.num_validation:]

    def GetValidationSet(self):
        return self.XDev, self.YDev

    def GetTestSet(self):
        return self.XTest, self.YTest

    # 获得批样本数据
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
