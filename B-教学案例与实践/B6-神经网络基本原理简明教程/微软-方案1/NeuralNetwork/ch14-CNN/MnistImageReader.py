# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding: utf-8

import numpy as np
import struct
from MiniFramework.DataReader import *

# XTrainRaw - train data, not normalized
# XTestRaw - test data, not normalized

# YTrainRaw - train label data, not normalized
# YTestRaw - test label data, not normalized

# X - XTrainSet + XDevSet
# XTrainSet - train data normalized, come from XTrainRaw
# XDevSet - validation data, normalized, come from X
# XTestSet - test data, normalized, come from XTestRaw

# Y - YTrainSet + YDevSet
# YTrainSet - train label data normalized, come from YTrainRaw (one-hot, or 0/1)
# YDevSet - validation label data, normalized, come from YTrain
# YTestSet - test label data, normalized, come from YTestRaw (one-hot or 0/1)


class MnistImageReader(DataReader):
    def __init__(self, train_image_file, train_label_file, test_image_file, test_label_file):
        self.train_image_file = train_image_file
        self.train_label_file = train_label_file
        self.test_image_file = test_image_file
        self.test_label_file = test_label_file
        self.num_example = 0
        self.num_feature = 0
        self.num_category = 0
        self.num_validation = 0
        self.num_test = 0
        self.num_train = 0

    def ReadData(self):
        self.XTrainRaw = self.__ReadImageFile(self.train_image_file)
        self.YTrainRaw = self.__ReadLabelFile(self.train_label_file)
        self.XTestRaw = self.__ReadImageFile(self.test_image_file)
        self.YTestRaw = self.__ReadLabelFile(self.test_label_file)
        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = len(np.unique(self.YTrainRaw))
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        self.num_validation = 0

    # output array: num_images * channel * 28 * 28
    # due to gray image instead of color, so channel = 1
    def __ReadImageFile(self, image_file_name):
        # header
        f = open(image_file_name, "rb")
        a = f.read(4)
        b = f.read(4)
        num_images = int.from_bytes(b, byteorder='big')
        c = f.read(4)
        num_rows = int.from_bytes(c, byteorder='big')
        d = f.read(4)
        num_cols = int.from_bytes(d, byteorder='big')
        # image data binary
        image_size = num_rows * num_cols    # 28x28=784
        fmt = '>' + str(image_size) + 'B'
        image_data = np.empty((num_images,1,num_rows,num_cols)) # N x 1 x 28 x 28
        for i in range(num_images):
            bin_data = f.read(image_size)   # read 784 byte data for one time
            unpacked_data = struct.unpack(fmt, bin_data)
            array_data = np.array(unpacked_data)
            array_data2 = array_data.reshape((1, num_rows, num_cols))
            image_data[i] = array_data2
        f.close()
        return image_data

    def __ReadLabelFile(self, lable_file_name):
        f = open(lable_file_name, "rb")
        f.read(4)
        a = f.read(4)
        num_labels = int.from_bytes(a, byteorder='big')

        fmt = '>B'
        label_data = np.zeros((num_labels,1))   # N x 1
        for i in range(num_labels):
            bin_data = f.read(1)
            unpacked_data = struct.unpack(fmt, bin_data)[0]
            label_data[i] = unpacked_data
        f.close()
        return label_data

    def Normalize(self):
        self.NormalizeX()
        self.NormalizeY()

    def NormalizeX(self):
        self.X = self.__NormalizeData(self.XTrainRaw).astype(np.float32)
        self.XTestSet = self.__NormalizeData(self.XTestRaw).astype(np.float32)

    def NormalizeY(self):
        self.Y = self.ToOneHot(self.YTrainRaw)
        # no need to OneHot test set, we only need to get [0~9] from this set, instead of onehot encoded
        self.YTestSet = self.YTestRaw

    def ToOneHot(self, dataSet):
        num = dataSet.shape[0]
        Y = np.zeros((num, self.num_category))
        for i in range(num):
            n = (int)(dataSet[i])
            Y[i,n] = 1
            # end if
        # end for
        return Y

    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape).astype(np.float32)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW

    # need explicitly call this function to generate validation set
    def GenerateDevSet(self, k = 10):
        self.num_validation = (int)(self.num_example / k)
        # dev set
        self.XDevSet = self.X[0:self.num_validation]
        self.YDevSet = self.Y[0:self.num_validation]
        # train set
        self.XTrainSet = self.X[self.num_validation:]
        self.YTrainSet = self.Y[self.num_validation:]
        
        self.num_train = self.num_example - self.num_validation

    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        if self.num_validation == 0:
            batch_X = self.X[start:end]
            batch_Y = self.Y[start:end]
        else:
            batch_X = self.XTrainSet[start:end]
            batch_Y = self.YTrainSet[start:end]
        return batch_X, batch_Y.T

    # recommend not use this function in DeepLearning
    def GetBatchValidationSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        if self.num_validation == 0:
            batch_X = self.X[start:end]
            batch_Y = self.Y[start:end]
        else:
            batch_X = self.XDevSet[start:end]
            batch_Y = self.YDevSet[start:end]
        return batch_X, batch_Y.T

    def GetBatchTestSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTestSet[start:end]
        batch_Y = self.YTestSet[start:end]
        return batch_X, batch_Y

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    # suggest to call this function for each epoch
    def Shuffle(self):
        if self.num_validation == 0:
            seed = np.random.randint(0,100)
            np.random.seed(seed)
            XP = np.random.permutation(self.X)
            np.random.seed(seed)
            YP = np.random.permutation(self.Y)
            self.X = XP
            self.Y = YP
            return self.X, self.Y
        else:
            seed = np.random.randint(0,100)
            np.random.seed(seed)
            XP = np.random.permutation(self.XTrainSet)
            np.random.seed(seed)
            YP = np.random.permutation(self.YTrainSet)
            self.XTrainSet = XP
            self.YTrainSet = YP
            return self.XTrainSet, self.YTrainSet

