# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding: utf-8

import numpy as np
import struct
from HelperClass2.DataReader_2_0 import *

train_image_file = '../../Data/train-images-10'
train_label_file = '../../Data/train-labels-10'
test_image_file = '../../Data/test-images-10'
test_label_file = '../../Data/test-labels-10'

class MnistImageDataReader(DataReader_2_0):
    # mode: "image"=Nx1x28x28,  "vector"=1x784
    def __init__(self, mode="image"):
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
        self.mode = mode    # image or vector

    def ReadLessData(self, count):
        self.XTrainRaw = self.__ReadImageFile(self.train_image_file)
        self.YTrainRaw = self.__ReadLabelFile(self.train_label_file)
        self.XTestRaw = self.__ReadImageFile(self.test_image_file)
        self.YTestRaw = self.__ReadLabelFile(self.test_label_file)

        self.XTrainRaw = self.XTrainRaw[0:count]
        self.YTrainRaw = self.YTrainRaw[0:count]

        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = (np.unique(self.YTrainRaw)).shape[0]
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        if self.mode == "vector":
            self.num_feature = 784
        self.num_validation = 0


    def ReadData(self):
        self.XTrainRaw = self.__ReadImageFile(self.train_image_file)
        self.YTrainRaw = self.__ReadLabelFile(self.train_label_file)
        self.XTestRaw = self.__ReadImageFile(self.test_image_file)
        self.YTestRaw = self.__ReadLabelFile(self.test_label_file)
        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = (np.unique(self.YTrainRaw)).shape[0]
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        if self.mode == "vector":
            self.num_feature = 784
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
        # end for
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

    def NormalizeX(self):
        self.XTrain = self.__NormalizeData(self.XTrainRaw)
        self.XTest = self.__NormalizeData(self.XTestRaw)

    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW

    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        if self.num_validation == 0:
            batch_X = self.XTrain[start:end]
            batch_Y = self.YTrain[start:end]
        else:
            batch_X = self.XTrain[start:end]
            batch_Y = self.YTrain[start:end]
        # end if

        if self.mode == "vector":
            return batch_X.reshape(-1, 784), batch_Y
        elif self.mode == "image":
            return batch_X, batch_Y

    # recommend not use this function in DeepLearning
    def GetValidationSet(self):
        batch_X = self.XDev
        batch_Y = self.YDev
        if self.mode == "vector":
            return batch_X.reshape(self.num_validation, -1), batch_Y
        elif self.mode == "image":
            return batch_X, batch_Y

    def GetTestSet(self):
        if self.mode == "vector":
            return self.XTest.reshape(self.num_test,-1), self.YTest
        elif self.mode == "image":
            return self.XTest, self.YTest

    def GetBatchTestSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTest[start:end]
        batch_Y = self.YTest[start:end]

        if self.mode == "vector":
            return batch_X.reshape(batch_size, -1), batch_Y
        elif self.mode == "image":
            return batch_X, batch_Y

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    # suggest to call this function for each epoch
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
        return self.XTrain, self.YTrain

