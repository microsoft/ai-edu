# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
from MiniFramework.DataReader_2_0 import *
from MiniFramework.EnumDef_6_0 import *
import random

train_file = '../../Data/ch19_pm25_train.npz'
test_file = '../../Data/ch19_pm25_test.npz'

"""
field: year, month, day, hour, dew, temp, air_press, wind_direction, wind_speed
"""

class PM25DataReader(DataReader_2_0):
    def __init__(self, mode):
        self.mode = mode    # mode = NetType.Fitting : NetType.MulitpleClassifier
        self.train_file_name = train_file
        self.test_file_name = test_file
        self.num_example = 0
        self.num_feature = 0
        self.num_category = 0
        self.num_validation = 0
        self.num_test = 0
        self.num_train = 0

    def ReadData(self, timestep = 24):
        super().ReadData()
        if (self.mode == NetType.Fitting):
            self.YTrainRaw = self.YTrainRaw[:,0]
            self.YTestRaw = self.YTestRaw[:,0]
        elif (self.mode == NetType.MultipleClassifier):
            self.YTrainRaw = self.YTrainRaw[:,1].reshape(-1,1)
            self.YTestRaw = self.YTestRaw[:,1].reshape(-1,1)
            self.num_category = len(npy.unique(self.YTrainRaw))
    
        self.num_example = self.YTrainRaw.shape[0]
        self.num_train = self.num_example - timestep
        tmp_x = np.empty((self.num_train, timestep, self.num_feature))
        tmp_y = np.empty((self.num_train, 1))
        for i in range(self.num_train):
            for j in range(timestep):
                tmp_x[i,j] = self.XTrainRaw[i+j]
            tmp_y[i] = self.YTrainRaw[i + timestep]

    def GetBatchTrainSamples(self, batch_size, time_step):
        start = random.randint(0, self.num_train - time_step)
        end = start + batch_size
        batch_X = np.empty((batch_size, time_step, self.num_feature))


        batch_X = self.XTrain[start:end]
        batch_Y = self.YTrain[start:end]
        return batch_X, batch_Y

