# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

train_data_name = "../../data/ch19.train_echo.npz"
test_data_name = "../../data/ch19.test_echo.npz"

def create_data(count, tt, filename):
    S = np.random.rand(count + tt - 1, 1)
    # count个样本, 2个timestep, 1个特征
    X = np.zeros((count, 2, 1))
    Y = np.zeros((count, 2))
    for i in range(count):
        for j in range(tt):
            X[i,j,0] = S[i+j]
        #end for
        Y[i,0] = 0
        Y[i,1] = S[i]
    #end for
    np.savez(filename, data=X, label=Y)

if __name__=='__main__':
    create_data(200, 2, train_data_name)
    create_data(50, 2, test_data_name)
    print("done")
