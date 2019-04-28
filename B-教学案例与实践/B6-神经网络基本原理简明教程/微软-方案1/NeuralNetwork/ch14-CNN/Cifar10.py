# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

# coding: utf-8

import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    dict = unpickle("c:\\MyDocument\\cifar-10-batches-py\\data_batch_1")
    print(dict)
