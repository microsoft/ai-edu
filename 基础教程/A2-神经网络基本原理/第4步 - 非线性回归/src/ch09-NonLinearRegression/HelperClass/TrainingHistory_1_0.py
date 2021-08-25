# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class TrainingHistory_1_0(object):
    def __init__(self):
        self.iteration = []
        self.loss_history = []

    def AddLossHistory(self, iteration, loss):
        self.iteration.append(iteration)
        self.loss_history.append(loss)

    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.iteration, self.loss_history)
        title = params.toString()
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
        return title

    def GetLast(self):
        count = len(self.loss_history)
        return self.loss_history[count-1], self.w_history[count-1], self.b_history[count-1]
# end class
