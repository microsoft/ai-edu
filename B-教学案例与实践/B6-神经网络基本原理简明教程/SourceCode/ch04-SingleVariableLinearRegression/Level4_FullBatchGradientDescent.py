# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# multiple iteration, loss calculation, stop condition, predication
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Level4_Base import *

if __name__ == '__main__':
    
    params = CParameters(eta=0.5, max_epoch=1000, batch_size=-1, eps = 0.0215)
    train(params)
