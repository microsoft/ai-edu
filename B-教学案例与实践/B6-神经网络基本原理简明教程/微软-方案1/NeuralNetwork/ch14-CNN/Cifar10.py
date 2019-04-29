# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

# coding: utf-8

import pickle

from CifarImageReader import *

file_1 = "..\\Data\\data_batch_1.bin"
file_2 = "..\\Data\\data_batch_2.bin"
file_3 = "..\\Data\\data_batch_3.bin"
file_4 = "..\\Data\\data_batch_4.bin"
file_5 = "..\\Data\\data_batch_5.bin"
test_file = "..\\Data\\test_batch.bin"

if __name__ == '__main__':
    dr = CifarImageReader(file_1, file_2, file_3, file_4, file_5, test_file)
    dr.ReadData()
    dr.HoldOut(10)
    print(dr.num_validation, dr.num_example, dr.num_test, dr.num_train)