# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class CData(object):
    def __init__(self, loss, dict_weights, epoch, iteration):
        self.loss = loss
        self.dict_weights = dict_weights
        self.epoch = epoch
        self.iteration = iteration
