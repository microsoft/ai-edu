# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from Level0_TwoLayerNet import *
from Activations import *
from WeightsBias import *

class TwoLayerFittingNet(TwoLayerNet):

    def ForwardCalculationBatch(self, batch_x, wbs):
        # layer 1
        Z1 = np.dot(wbs.W1, batch_x) + wbs.B1
        A1 = CSigmoid().forward(Z1)
        # layer 2
        Z2 = np.dot(wbs.W2, A1) + wbs.B2
        A2 = Z2
        # keep cache for backward
        dict_cache ={"Z2": Z2, "A1": A1, "A2": A2, "Output": A2}
        return dict_cache

# end class