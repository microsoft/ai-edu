# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

# coding: utf-8

import pickle

from CifarImageReader import *

file_1 = "C:\\MyDocument\\cifar-10-batches-bin\\data_batch_1.bin"
file_2 = "C:\\MyDocument\\cifar-10-batches-bin\\data_batch_2.bin"
file_3 = "C:\\MyDocument\\cifar-10-batches-bin\\data_batch_3.bin"
file_4 = "C:\\MyDocument\\cifar-10-batches-bin\\data_batch_4.bin"
file_5 = "C:\\MyDocument\\cifar-10-batches-bin\\data_batch_5.bin"

if __name__ == '__main__':
    dr = CifarImageReader(file_1, None, None, None, None, None)
    dr.ReadImageFile(file_1)