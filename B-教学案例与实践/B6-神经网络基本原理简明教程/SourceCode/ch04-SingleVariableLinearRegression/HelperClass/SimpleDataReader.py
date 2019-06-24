# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
from enum import Enum

file_name = "../../data/ch04.npz"

class YNormalizationMethod(Enum):
    Nothing = 0,
    Regression = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3

"""
X:
    x1: feature1 feature2 feature3...
    x2: feature1 feature2 feature3...
    x3: feature1 feature2 feature3...
    ......

Y:  [if regression, value]
    [if binary classification, 0/1]
    [if multiple classification, e.g. 4 category, one-hot]

"""

class SimpleDataReader(object):
    def __init__(self):
        self.train_file_name = file_name
        self.num_train = 0
        self.XTrain = None
        self.YTrain = None

    # read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XTrain = data["data"]
            self.YTrain = data["label"]
            self.num_train = self.XTrain.shape[0]
        else:
            raise Exception("Cannot find train file!!!")
        #end if

    # get batch training data
    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    # get batch training data
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain


    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
