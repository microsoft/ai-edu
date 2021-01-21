# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from MiniFramework.DataReader_2_0 import *
from MiniFramework.EnumDef_6_0 import *
from ExtendedDataReader.MnistImageDataReader import *

class MnistTimeStepDataReader(MnistImageDataReader):
    def GetBatchTrainSamples(self, batch_size, iteration):
        batch_X, batch_Y = super.GetBatchTrainSamples(batch_size, iteration)
        if self.mode == "vector":
            return batch_X.reshape(-1, 784), batch_Y
        elif self.mode == "image":
            return batch_X.reshape(-1,28,28), batch_Y
        