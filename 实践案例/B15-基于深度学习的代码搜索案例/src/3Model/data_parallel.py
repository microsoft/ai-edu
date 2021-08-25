# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import torch.nn as nn
from typing import List


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class NoDataParallel(nn.Module):
    def __init__(self, module):
        super(NoDataParallel, self).__init__()
        self.module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


def get_data_parallel(model: nn.Module, device_ids: List[int]) -> nn.Module:
    if device_ids:
        if -1 in device_ids:
            model = MyDataParallel(model)
        else:
            model = MyDataParallel(model, device_ids=device_ids)
    else:
        model = NoDataParallel(model)
    return model
