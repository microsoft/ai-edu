# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

x_data_name = "HousePriceXData.npy"
y_data_name = "HousePriceYData.npy"

# y = w1*x1 + w2*x2 + w3*x3 + b
# W1 = 朝向：1,2,3,4 = N,W,E,S
# W2 = 位置几环：2,3,4,5,6
# W3 = 面积:平米
def TargetFunction(x1,x2,x3):
    w1,w2,w3,b = 2,10,5,10
    return w1*x1 + w2*(10-x2) + w3*x3 + b

def CreateSampleData(m):
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
    else:
        # [1,4] 朝向
        X0 = np.random.randint(1,5,m)
        # [2,6] 几环
        X1 = np.random.randint(2,7,m)
        # [40,120] 面积
        X2 = np.random.randint(40,120,m)
        XData = np.zeros((3,m))
        XData[0] = X0
        XData[1] = X1
        XData[2] = X2
        Y = TargetFunction(X0,X1,X2)
        noise = 20
        Noise = np.random.randint(1,noise,(1,m)) - noise/2
        YData = Y.reshape(1,m)
        np.save("HousePriceXData.npy", XData)
        np.save("HousePriceYData.npy", YData)
    return XData, YData

CreateSampleData(1000)