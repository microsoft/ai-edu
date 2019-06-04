# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding: utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt

from MiniFramework.DataReader import *

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
        self.num_train = self.num_example = self.XTrainRaw.shape[0]
        self.num_test = self.XTestRaw.shape[0]
        self.num_feature = 1024

    def __ReadTestFile(self):
        self.XTestRaw, self.YTestRaw = self.__ReadSingleDataFile(self.test_file)

    def __ReadTrainFiles(self):
        self.XTrainRaw = None
        self.YTrainRaw = None
        for i in range(5):
            image_data_single, label_data_single = self.__ReadSingleDataFile(self.train_file[i])
            if self.XTrainRaw is None:
                self.XTrainRaw = image_data_single
                self.YTrainRaw = label_data_single
            else:
                self.XTrainRaw = np.vstack((self.XTrainRaw, image_data_single))
                self.YTrainRaw = np.vstack((self.YTrainRaw, label_data_single))
            #end if
        #end for

    # output array: num_images * channel * 28 * 28
    # 3 color, so channel = 3
    def __ReadSingleDataFile(self, image_file_name):
        image_data = np.empty((10000,1024)).astype(np.float32)
        label_data = np.zeros((10000,10))
        f = open(image_file_name, "rb")
        for i in range(10000):
            a = f.read(1)
            label = int.from_bytes(a, byteorder='big')
            label_data[i,label] = 1
            color_data =np.empty((3,32,32))
            for j in range(3):
                b = f.read(1024)
                fmt = '=' + str(1024) + 'B'
                unpacked_data = struct.unpack(fmt, b)
                array_data = np.array(unpacked_data)
                array_data2 = array_data.reshape(32,32)
                color_data[j] = array_data2
            #end for

            #plt.imshow(color_data.transpose(1,2,0))
            #plt.show()

            gray_data = np.dot([0.299,0.587,0.114], color_data.reshape(3,-1)).reshape(1,1024)
            image_data[i] = gray_data
            #plt.imshow(gray_data)
            #plt.show()

        #end for
        f.close()
        return image_data.astype(np.float32), label_data


