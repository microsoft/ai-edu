# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from enum import Enum

class LayerTypes(Enum):
    FC = 0      # full connection
    CV = 1      # convalution
    PL = 2      # pooling
    BN = 3      # batch normalization
    DP = 4      # dropout


class CLayer(object):
    def __init__(self, layer_type):
        self.layer_type = layer_type

    def train(self, input, train=True):
        pass

    def update(self):
        pass

    def save_parameters(self, name):
        pass

    def load_parameters(self, name):
        pass

class LayerIndexFlags(Enum):
    SingleLayer = 0
    FirstLayer = 1
    LastLayer = -1
    MiddleLayer = 2

