# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import torch
import os
from typing import Any


def load_epoch(model_path: str, epoch: int) -> Any:
    print('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(model_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')
