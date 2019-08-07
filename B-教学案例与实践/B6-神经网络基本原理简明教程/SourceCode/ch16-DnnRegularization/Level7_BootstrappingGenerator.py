# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import os

from ExtendedDataReader.MnistImageDataReader import *

def GenerateDataSet(subfolder, count=10):
    isExists = os.path.exists(subfolder)
    if not isExists:
        os.makedirs(subfolder)

    mdr = MnistImageDataReader("vector")
    mdr.ReadLessData(1000)
    
    for i in range(count):
        X = np.zeros_like(mdr.XTrainRaw)
        Y = np.zeros_like(mdr.YTrainRaw)
        list = np.random.choice(1000,1000)
        k=0
        for j in list:
            X[k] = mdr.XTrainRaw[j]
            Y[k] = mdr.YTrainRaw[j]
            k = k+1
        # end for
        np.savez(subfolder + "/" + str(i) + ".npz", data=X, label=Y)
    # end for

if __name__=="__main__":
    GenerateDataSet("ensemble", 9)
  