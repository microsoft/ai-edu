# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

train_file = '../../Data/ch19_pm25_train.npz'
test_file = '../../Data/ch19_pm25_test.npz'


class PM25DataReader(DataReader_2_0):
    # mode = "regression" : "classification"
    def __init__(self, mode):
        self.mode = mode
        self.train_file = train_file
        self.test_file = test_file
        self.num_example = 0
        self.num_feature = 0
        self.num_category = 0
        self.num_validation = 0
        self.num_test = 0
        self.num_train = 0
        self.mode = mode    # image or vector

    def ReadData(self):
        self.XTrainRaw = self.ReadImageFile(self.train_image_file)
        self.YTrainRaw = self.ReadLabelFile(self.train_label_file)
        self.XTestRaw = self.ReadImageFile(self.test_image_file)
        self.YTestRaw = self.ReadLabelFile(self.test_label_file)
        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = (np.unique(self.YTrainRaw)).shape[0]
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        if self.mode == "vector":
            self.num_feature = 784
        self.num_validation = 0
