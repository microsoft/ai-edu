# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 1.0
- crossentropy for binary classifier (logistic)
- crossentropy for binary classifier (tanh)
"""

import numpy as np

from HelperClass.EnumDef_1_0 import *

class LossFunction_1_0(object):
    def __init__(self, net_type):
        self.net_type = net_type
    # end def

    # fcFunc: feed forward calculation
    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.BinaryTanh:
            loss = self.CE2_tanh(A, Y, m)
        #end if
        return loss
    # end def

    # for binary classifier
    def CE2(self, A, Y, count):
        p1 = 1 - Y
        p2 = np.log(1 - A)
        p3 = np.log(A)

        p4 = np.multiply(p1 ,p2)
        p5 = np.multiply(Y, p3)

        LOSS = np.sum(-(p4 + p5))  #binary classification
        loss = LOSS / count
        return loss
    # end def

    # for binary tanh classifier
    def CE2_tanh(self, A, Y, count):
        #p = (1-Y) * np.log(1-A) + (1+Y) * np.log(1+A)
        p = (1-Y) * np.log((1-A)/2) + (1+Y) * np.log((1+A)/2)
        LOSS = np.sum(-p)
        loss = LOSS / count
        return loss
    # end def
