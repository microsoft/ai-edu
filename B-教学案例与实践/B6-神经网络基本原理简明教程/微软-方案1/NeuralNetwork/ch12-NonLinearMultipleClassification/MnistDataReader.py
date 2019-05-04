# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import struct
from DataReader import *

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


class MnistDataReader(DataReader):
    def __init__(self, train_image_file, train_label_file, test_image_file, test_label_file):
        self.train_image_file = train_image_file
        self.train_label_file = train_label_file
        self.test_image_file = test_image_file
        self.test_label_file = test_label_file
        self.num_example = 0
        self.num_feature = 0
        self.num_category = 0
        self.validation_size = 0

    def ReadData(self):
        self.XTrainRaw = self.__ReadImageFile(self.train_image_file)
        self.YTrainRaw = self.__ReadLabelFile(self.train_label_file)
        self.XTestRaw = self.__ReadImageFile(self.test_image_file)
        self.YTestRaw = self.__ReadLabelFile(self.test_label_file)
        self.num_example = self.XTrainRaw.shape[1]
        self.num_feature = self.XTrainRaw.shape[0]
        self.num_category = len(np.unique(self.YTrainRaw))

    # output array: 768 x num_images
    def __ReadImageFile(self, image_file_name):
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

    def __ReadLabelFile(self, lable_file_name):
        f = open(lable_file_name, "rb")
        f.read(4)
        a = f.read(4)
        num_labels = int.from_bytes(a, byteorder='big')

        fmt = '>B'
        label_data = np.zeros((1, num_labels))   # 1 x M
        for i in range(num_labels):
            bin_data = f.read(1)
            unpacked_data = struct.unpack(fmt, bin_data)[0]
            label_data[0,i] = unpacked_data
        f.close()
        return label_data

    def Normalize(self):
        self.NormalizeX()
        self.NormalizeY()

    def NormalizeX(self):
        self.X = self.__NormalizeData(self.XTrainRaw)
        self.XTestSet = self.__NormalizeData(self.XTestRaw)

    def NormalizeY(self):
        self.Y = self.ToOneHot(self.YTrainRaw)
        self.YTestSet = self.ToOneHot(self.YTestRaw)

    def ToOneHot(self, dataSet):
        num = dataSet.shape[1]
        Y = np.zeros((self.num_category, num))
        for i in range(num):
            n = (int)(dataSet[0,i])
            Y[n,i] = 1
            # end if
        # end for
        return Y

    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW

    # need explicitly call this function to generate validation set
    def GenerateDevSet(self, k = 10):
        self.validation_size = (int)(self.num_example / k)
        self.XDevSet = self.X[:,0:validation_size]
        self.XTrainSet = self.X[:,validation_size:]
        self.YDevSet = self.Y[:,0:validation_size]
        self.YTrainSet = self.X[:,validation_size:]

    def GetBatchSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        if self.validation_size == 0:
            batch_X = self.X[0:self.num_feature, start:end].reshape(self.num_feature, batch_size)
            batch_Y = self.Y[:, start:end].reshape(-1, batch_size)
        else:
            batch_X = self.XTrainSet[0:self.num_feature, start:end].reshape(self.num_feature, batch_size)
            batch_Y = self.YTrainSet[:, start:end].reshape(-1, batch_size)
        return batch_X, batch_Y

        # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.X.T)
        np.random.seed(seed)
        YP = np.random.permutation(self.Y.T)
        self.X = XP.T
        self.Y = YP.T
        return self.X, self.Y

