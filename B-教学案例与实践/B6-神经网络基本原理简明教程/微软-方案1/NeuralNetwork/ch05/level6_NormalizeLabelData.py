# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from level5_NormalizePredicateData import *

def NormalizeXYData(XData, YData):
    X, X_norm = NormalizeData(XData)
    Y, Y_norm = NormalizeData(YData)
    return X, X_norm, Y, Y_norm

# try to give the answer for the price of west(2)，5th ring(5)，93m2
def PredicateTest(W, B, X_norm, Y_norm):
    xt = np.array([2,5,93]).reshape(3,1)
    xt_new = NormalizePredicateData(xt, X_norm)
    z = ForwardCalculationBatch(W, B, xt_new)
    zz = z * Y_norm[1,0] + Y_norm[0,0]
    return zz

# main
if __name__ == '__main__':
    # hyper parameters
    # SGD, MiniBatch, FullBatch
    method = "FullBatch"
    # read data
    raw_X, raw_Y = ReadData()
    X, X_norm, Y, Y_norm = NormalizeXYData(raw_X, raw_Y)
    w, b = train(method, X, Y)
    z = PredicateTest(w, b, X_norm, Y_norm)
    print("z=", z)
