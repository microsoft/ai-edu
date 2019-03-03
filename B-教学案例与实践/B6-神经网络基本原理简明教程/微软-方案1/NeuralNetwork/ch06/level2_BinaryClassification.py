# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "Pollution2CategoryX.dat"
y_data_name = "Pollution2CategoryY.dat"

# binary classification, modify Xdata to 2 lines
def LoadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)

        return XData,YData
    
    return None,None


def LoadData2():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)

        return XData,YData
    
    return None,None



def ShowResult(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 0 and Y[1,i]==1:
            plt.scatter(X[0,i], X[1,i], c='b')
        else:
            plt.scatter(X[0,i], X[1,i], c='g')
    plt.show()

if __name__ == '__main__':
    XData, Y = LoadData2()
    print(XData.shape, Y.shape)
    ShowResult(XData,Y)
    '''
    count = (int)(200 - Y[2,:].sum())
    X_new = np.zeros((2,count))
    Y_new = np.zeros((2,count))
    j = 0
    for i in range(200):
        if Y[0,i] == 1 or Y[1,i] == 1:
            X_new[:,j] = XData[:,i]
            Y_new[0:2,j] = Y[0:2,i]
            j = j + 1

    ShowResult(X_new,Y_new)
    np.save("x", X_new)
    np.save("y", Y_new)
    '''


