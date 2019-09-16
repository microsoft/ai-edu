# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as npy
from pathlib import Path
from MiniFramework.EnumDef_6_0 import *
from MiniFramework.DataReader_2_0 import *

class DataReader_3_0(DataReader_2_0):
    def GenerateRnnSamples(self, tt):
        self.num_train = self.num_train - tt + 1
        self.XTrain = np.empty((self.num_train, tt))
        self.YTrain = np.empty((self.num_train, tt))
        for i in range(self.num_train):
            for j in range(tt):
                self.XTrain[i,j] = self.XTrainRaw[i+j]
                self.YTrain[i,j] = self.YTrainRaw[i+j]

        self.num_test = self.num_test - tt + 1
        self.XTest = np.empty((self.num_test, tt))
        self.YTest = np.empty((self.num_test, tt))
        for i in range(self.num_test):
            for j in range(tt):
                self.XTest[i,j] = self.XTestRaw[i+j]
                self.YTest[i,j] = self.YTestRaw[i+j]
