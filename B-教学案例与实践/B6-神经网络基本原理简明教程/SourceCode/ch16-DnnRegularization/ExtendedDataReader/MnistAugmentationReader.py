# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from ExtendedDataReader.MnistImageDataReader import *

test_image_file = '../../Data/test-images-10'
test_label_file = '../../Data/test-labels-10'

class MnistAugmentationReader(MnistImageDataReader):
    def ReadData(self):
        data = np.load("augmentation/data.npz")
        image = data["data"] 
        label = data["label"]
        assert(image.shape[0] == label.shape[0])
        count = image.shape[0]
        self.XTrainRaw = image.reshape(count, 1, 28, 28)
        self.YTrainRaw = label.reshape(count, 1)
        self.XTestRaw = self.ReadImageFile(self.test_image_file)
        self.YTestRaw = self.ReadLabelFile(self.test_label_file)
        self.num_example = self.XTrainRaw.shape[0]
        self.num_category = len(np.unique(self.YTrainRaw))
        self.num_test = self.XTestRaw.shape[0]
        self.num_train = self.num_example
        if self.mode == "vector":
            self.num_feature = 784
        self.num_validation = 0
