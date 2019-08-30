# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import struct
from MiniFramework.DataReader_2_0 import *

class GeometryDataReader(DataReader_2_0):
    # mode = 'image' or 'vector' 
    def __init__(self, train_file, test_file, mode):
        super(GeometryDataReader, self).__init__(train_file, test_file)
        self.mode = mode

    def ReadData(self):
        super().ReadData()
        if self.mode == 'vector':
            (N,C,H,W) = self.XTrainRaw.shape
            new_data = np.empty((N,H,W))
            if C == 3: # color
                for i in range(N):
                    new_data[i] = np.dot([0.299,0.587,0.114], self.XTrainRaw[i].reshape(3,-1)).reshape(1,784)
            elif C == 1: # gray
                new_data[i] = self.XTrainRaw[i].reshape(1,784)
            #end if
            self.num_feature = 784

    def NormalizeX(self):
        self.XTrain = self.__NormalizeData(self.XTrainRaw)
        self.XTest = self.__NormalizeData(self.XTestRaw)

    def __NormalizeData(self, XRawData):
        X_NEW = np.zeros(XRawData.shape)
        x_max = np.max(XRawData)
        x_min = np.min(XRawData)
        X_NEW = (XRawData - x_min)/(x_max-x_min)
        return X_NEW
