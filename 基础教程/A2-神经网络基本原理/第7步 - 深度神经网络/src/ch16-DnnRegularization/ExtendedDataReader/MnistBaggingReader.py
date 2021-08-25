# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from ExtendedDataReader.MnistImageDataReader import *

train_image_file_temp = 'ensemble/{0}.npz'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

class MnistBaggingReader(MnistImageDataReader):
    def ReadData(self, index):
        train_image_file = str.format(train_image_file_temp, index)
        data = np.load(train_image_file)
        self.XTrainRaw = data["data"]
        self.YTrainRaw = data["label"]
        self.XTestRaw = self.ReadImageFile(self.test_image_file)
        self.YTestRaw = self.ReadLabelFile(self.test_label_file)
        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = len(np.unique(self.YTrainRaw))
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        if self.mode == "vector":
            self.num_feature = 784
        self.num_validation = 0


