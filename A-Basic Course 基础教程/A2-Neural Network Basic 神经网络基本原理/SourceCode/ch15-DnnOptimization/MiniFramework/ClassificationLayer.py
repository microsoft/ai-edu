# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

from MiniFramework.Layer import *

# sigmoid and softmax
class ClassificationLayer(CLayer):
    def __init__(self, classifier):
        self.classifier = classifier

    def forward(self, input, train=True):
        self.x = input
        self.a = self.classifier.forward(self.x)
        return self.a

    # 对分类函数的求导已经和损失函数合并计算了，所以不需要再做，直接回传误差给上一层
    def backward(self, delta_in, flag):
        dZ = delta_in
        return dZ

class CClassifier(object):
    def forward(self, z):
        pass

class Softmax(CClassifier):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a

class Logistic(CClassifier):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a
