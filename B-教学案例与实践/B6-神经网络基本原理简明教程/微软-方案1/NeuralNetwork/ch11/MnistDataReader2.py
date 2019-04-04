# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import struct
from Level4.DataReader import *

class MnistDataReader(DataReader):
    def __init__(self, train_image_file, train_label_file, test_image_file, test_label_file):
        self.train_image_file = train_image_file
        self.train_label_file = train_label_file
        self.test_image_file = test_image_file
        self.test_label_file = test_label_file
        self.num_example = -1
        self.num_feature = -1
        self.num_category = -1

    def ReadFile(self):
        self.__ReadTrainImageFile()
        self.__ReadTrainLabelFile()
        self.__ReadTestImageFile()
        self.__ReadTestLabelFile()

    def __ReadTrainImageFile(self):
        self.XRawTrainData = self.__ReadImageFile(self.train_image_file)
        self.num_example = self.XRawTrainData.shape[1]
        self.num_feature = self.XRawTrainData.shape[0]

    def __ReadTrainLabelFile(self):
        self.num_category = 10
        self.Y = self.__ReadLabelFile(self.train_label_file, 10)

    def __ReadTestImageFile(self):
        self.XRawTestData = self.__ReadImageFile(self.test_image_file)

    def __ReadTestLabelFile(self):
        self.YTestData = self.__ReadLabelFile(self.test_label_file, 10)

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

    def __ReadLabelFile(self, lable_file_name, num_output):
        f = open(lable_file_name, "rb")
        f.read(4)
        a = f.read(4)
        num_labels = int.from_bytes(a, byteorder='big')

        fmt = '>B'
        label_data = np.zeros((num_output, num_labels))   # 10 x M
        for i in range(num_labels):
            bin_data = f.read(1)
            unpacked_data = struct.unpack(fmt, bin_data)[0]
            label_data[unpacked_data,i] = 1
        f.close()
        return label_data

    def NormalizeXData(self):
        self.X = self.__NormalizeData(self.XRawTrainData)
        self.XTestData = self.__NormalizeData(self.XRawTestData)

    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW

    def GetBatchSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.X[0:self.num_feature, start:end].reshape(self.num_feature, batch_size)
        batch_Y = self.Y[:, start:end].reshape(-1, batch_size)
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
