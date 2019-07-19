# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 4.0
what's new?
- add OptimizerName
"""

from enum import Enum

class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2,
    MSRA = 3

class XCoordinate(Enum):
    Nothing = 0,
    Iteration = 1,
    Epoch = 2

class OptimizerName(Enum):
    SGD = 0,
    Momentum = 1,
    Nag = 2,
    AdaGrad = 3,
    AdaDelta = 4,
    RMSProp = 5,
    Adam = 6
