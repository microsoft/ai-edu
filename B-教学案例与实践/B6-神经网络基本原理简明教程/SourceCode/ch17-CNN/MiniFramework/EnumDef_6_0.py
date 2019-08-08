# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 4.1
what's new:
- add regularization enum
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

class StopCondition(Enum):
    Nothing = 0,    # reach the max_epoch then stop
    StopLoss = 1,   # reach specified loss value then stop
    StopDiff = 2,   # reach specified abs(curr_loss - prev_loss)

class Stopper(object):
    def __init__(self, sc, sv):
        self.stop_condition = sc
        self.stop_value = sv

class RegularMethod(Enum):
    Nothing = 0,
    L1 = 1,
    L2 = 2,
    EarlyStop = 3

class PoolingTypes(Enum):
    MAX = 0,
    MEAN = 1,
