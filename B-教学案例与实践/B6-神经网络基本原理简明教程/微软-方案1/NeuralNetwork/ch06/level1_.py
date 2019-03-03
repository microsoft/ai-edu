# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "PollutionCategoryXData.dat"
y_data_name = "PollutionCategoryYData.dat"

def LoadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
        return XData,YData
    
    return None,None

