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


class Mnist16DataReader(DataReader):
    def __init__(self, train_image_10, train_label_10, train_image_6, train_label_6, test_image_10, test_label_10, test_image_6, test_label_6):
        self.train_image_10 = train_image_10
        self.train_label_10 = train_label_10
        self.train_image_6 = train_image_6
        self.train_label_6 = train_label_6
        self.test_image_10 = test_image_10
        self.test_label_10 = test_label_10
        self.test_image_6 = test_image_6
        self.test_label_6 = test_label_6
        self.num_example = 0
        self.num_feature = 784
        self.num_category = 16
        self.validation_size = 0

    def ReadData(self):
        self.X = np.concatenate((self.train_image_10, self.train_image_6), axis=1)
        self.Y = self.ToOneHot(self.train_label_10, self.train_label_6)
        self.XTestSet = np.concatenate((self.test_image_10, self.test_image_6), axis=1)
        self.YTestSet = self.ToOneHot(self.test_label_10, self.test_label_6)
        self.num_example = self.X.shape[1]
        self.num_feature = self.X.shape[0]
        self.num_category = self.Y.shape[0]

    def ToOneHot(self, d10, d6):
        n1 = d10.shape[1]
        n2 = d6.shape[1]
        Y = np.zeros((16, n1 + n2))
        for i in range(n1):
            Y[0:10,i] = d10[:,i]
        # end for
        for j in range(n2):
            Y[10:16, j + n1] = d6[:, j]
        # end for
        return Y

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

