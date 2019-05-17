# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from enum import Enum

class LayerTypes(Enum):
    FC = 0      # full connection
    CONV = 1    # convalution
    POOL = 2    # pooling


class CLayer(object):
    def __init__(self, layer_type):
        self.layer_type = layer_type

    def update(self):
        return

class LayerIndexFlags(Enum):
    SingleLayer = 0
    FirstLayer = 1
    LastLayer = -1
    MiddleLayer = 2

