# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from numba import jitclass
from numba import float32

spec = [
    ('value', float32[:]),
    ('array', float32[:]),
]

@jitclass(spec)
class ConvWeightsBias(object):
    def __init__(self, value):
        self.value = value
        self.array = np.random.random(size=(10,10))

    def Do(self, val):
        return self.array * val

if __name__ == '__main__':
    c = ConvWeightsBias(np.random.random(size=(10,10)))
    a = c.Do(np.random.random(size=(10,10)))
    print(a)