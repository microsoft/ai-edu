
import numpy as np

from MiniFramework.ConvWeightsBias import *

import numpy as np
from numba import jitclass          # import the decorator
from numba import int32, float64    # import the types

spec = [
#    ('value', float64),               # a simple scalar field
    #('array', float64),          # an array field
]


@jitclass(spec)
class Bag(object):
    def __init__(self):
        pass
 #       self.value = 1
        #self.array = np.random.random()

    def increment(self, val1, val2):
        return val1 * val2

array = np.random.random()



mybag = Bag()
print('isinstance(mybag, Bag)', isinstance(mybag, Bag))
#print('mybag.value', mybag.value)


print('mybag.increment(3)', mybag.increment(np.random.random(size=(10,10)),np.random.random(size=(10,10))))
