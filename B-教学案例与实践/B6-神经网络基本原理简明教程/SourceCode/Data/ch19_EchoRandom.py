# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

train_data_name = "../../data/ch19.train_echo.npz"
test_data_name = "../../data/ch19.test_echo.npz"

def create_echo_random_data(train_count, test_count):
    X = np.random.rand(train_count,1)
    Y = np.zeros(X.shape)
    Y[1:train_count] = X[0:train_count-1]
    np.savez(train_data_name, data=X, label=Y)

    X = np.random.rand(test_count,1)
    Y = np.zeros(X.shape)
    Y[1:test_count] = X[0:test_count-1]
    np.savez(test_data_name, data=X, label=Y)

if __name__=='__main__':
    create_echo_random_data(200,50)
    print("done")
