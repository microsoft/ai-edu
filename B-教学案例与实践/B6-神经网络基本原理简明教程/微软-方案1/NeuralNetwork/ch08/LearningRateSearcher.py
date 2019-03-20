# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np


class CLooper(object):
    # e.g. (0.01, 0.005, 100) means start from 0.01, every 100 iteration will add 0.005 until 0.01*10 (=0.1)
    # in another word: 0.01,0.015,0.02,0.025,...0.09,0.095,0.1 (stop)
    def __init__(self, start, step, loop):
        self.start = start
        self.stop = start * 10
        self.step = step
        self.loop = loop

    # 把Looper展开成两个一维数组,后面便于计算
    def to_array(self):
        lrs = []
        loops = []
        i = 0
        while(True):
            lr = self.start + i * self.step
            if lr > self.stop:
                break
            else:
                lrs.append(lr)
                loops.append(self.loop)
                i = i + 1
            # end if
        #end while
        return lrs, loops

class CLearningRateSearcher(object):
    def __init__(self):
        self.learningRates = []
        self.loopCount = []
        self.searchIndex = -1
        
        # 记录训练过程中当前的lr对应的loss值
        self.loss_history = []
        self.lr_history = []

    def addLooper(self,looper):
        self.searchIndex = 0
        lrs, loops = looper.to_array()
        self.learningRates.extend(lrs)
        self.loopCount.extend(loops)

    def getNextLearningRate(self):
        if self.searchIndex == -1 or self.searchIndex >= len(self.learningRates):
            return None,None
        #end if
        
        lr = self.learningRates[self.searchIndex]
        loop = self.loopCount[self.searchIndex]
        self.searchIndex = self.searchIndex + 1
        return lr, loop

    def addHistory(self, loss, lr):
        self.loss_history.append(loss)
        self.lr_history.append(lr)

# unit test
if __name__ == '__main__':
    lrSearcher = CLearningRateSearcher()
    looper = CLooper(0.001,0.003,2)
    lrSearcher.addLooper(looper)
    looper = CLooper(0.01,0.04,3)
    lrSearcher.addLooper(looper)

    while(True):
        lr,loop = lrSearcher.getNextLearningReate()
        if lr == None:
            break
        print(lr,loop)


