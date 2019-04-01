# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from Level4_LogicGateBase import *

# x1=0,x2=0,y=1
# x1=0,x2=1,y=0
# x1=1,x2=0,y=0
# x1=1,x2=1,y=0
def Read_NOR_Data():
    X = np.array([0,0,1,1,0,1,0,1]).reshape(2,4)
    Y = np.array([1,0,0,0]).reshape(1,4)
    return X,Y

def Test(W,B):
    n1 = input("input number one:")
    x1 = float(n1)
    n2 = input("input number two:")
    x2 = float(n2)
    a = ForwardCalculationBatch(W, B, np.array([x1,x2]).reshape(2,1))
    print(a)
    y = not(x1 or x2)
    if np.abs(a-y) < 1e-2:
        print("True")
    else:
        print("False")


if __name__ == '__main__':
    # SGD, MiniBatch, FullBatch
    # read data
    X,Y = Read_NOR_Data()
    W, B = train(X, Y, ForwardCalculationBatch, CheckLoss)

    print("w=",W)
    print("b=",B)
    ShowResult(W,B,X,Y,"NOR")
    # test
    while True:
        Test(W,B)
