# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding: utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt

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


class CifarImageReader(DataReader):
    def __init__(self, train_file_1, train_file_2, train_file_3, train_file_4, train_file_5, test_file):
        self.train_file = []
        self.train_file.append(train_file_1)
        self.train_file.append(train_file_2)
        self.train_file.append(train_file_3)
        self.train_file.append(train_file_4)
        self.train_file.append(train_file_5)
        self.test_file = test_file
        self.num_example = 0
        self.num_category = 10
        self.num_validation = 0
        self.num_test = 0
        self.num_train = 0

    def ReadData(self):
        self.__ReadTrainFiles()
        self.__ReadTestFile()
        self.num_train = self.num_example = self.X.shape[0]
        self.num_test = self.XTestSet.shape[0]

    def __ReadTestFile(self):
        self.XTestSet, self.YTestSet = self.__ReadSingleDataFile(self.test_file)

    def __ReadTrainFiles(self):
        self.X = None
        self.Y = None
        for i in range(5):
            image_data_single, label_data_single = self.__ReadSingleDataFile(self.train_file[i])
            if self.X is None:
                self.X = image_data_single
                self.Y = label_data_single
            else:
                self.X = np.vstack((self.X, image_data_single))
                self.Y = np.vstack((self.Y, label_data_single))
            #end if
        #end for

    # output array: num_images * channel * 28 * 28
    # 3 color, so channel = 3
    def __ReadSingleDataFile(self, image_file_name):
        image_data = np.empty((10000,3,32,32)).astype(np.float32)
        label_data = np.zeros((10000,10))
        f = open(image_file_name, "rb")
        for i in range(10000):
            a = f.read(1)
            label = int.from_bytes(a, byteorder='big')
            label_data[i,label] = 1
            for j in range(3):
                b = f.read(1024)
                fmt = '=' + str(1024) + 'B'
                unpacked_data = struct.unpack(fmt, b)
                array_data = np.array(unpacked_data)
                array_data2 = array_data.reshape(32,32)
                image_data[i,j] = array_data2/255
            #end for
#            plt.imshow(image_data[i].transpose(1,2,0))
#            plt.show()
        #end for
        f.close()
        return image_data.astype(np.float32), label_data

    # need explicitly call this function to generate validation set
    def HoldOut(self, k = 10):
        self.Shuffle()
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
        return batch_X, batch_Y.T

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

