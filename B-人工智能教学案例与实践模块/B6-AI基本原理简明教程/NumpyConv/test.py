from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import numpy as np
import os
import shutil
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from conv2d import Cconv2d
from pool import Cpool
from relu import Crelu
from softmax import Csoftmax
from fc import Cfc
from dropout import Cdropout
import json

NUM_LABELS = 2

class Cmodel(object):
    def __init__(self, train_size):
        """The Model definition."""
        self.conv1 = Cconv2d([train_size, 28, 28, 1], 5, 32)
        self.relu1 = Crelu(self.conv1.outputShape)
        self.pool1 = Cpool(self.relu1.shape)
        self.conv2 = Cconv2d(self.pool1.outputShape, 5, 64)
        self.relu2 = Crelu(self.conv2.outputShape)
        self.pool2 = Cpool(self.relu2.shape)
        self.fc1 = Cfc(self.pool2.outputShape, 512)
        self.relu3 = Crelu(self.fc1.outputShape)
        self.dropout = Cdropout(self.relu3.shape, prob=0.5)
        self.fc2 = Cfc(self.dropout.outputShape, NUM_LABELS)
        self.softmax = Csoftmax([train_size, NUM_LABELS])
               
        self.conv1.weights = np.load("D:\\practice\\NumpyConv\\conv1_w.npy")
        self.conv1.bias = np.load("D:\\practice\\NumpyConv\\conv1_b.npy")
        self.conv2.weights = np.load("D:\\practice\\NumpyConv\\conv2_w.npy")
        self.conv2.bias = np.load("D:\\practice\\NumpyConv\\conv2_b.npy")
        self.fc1.weights = np.load("D:\\practice\\NumpyConv\\fc1_w.npy")
        self.fc1.bias = np.load("D:\\practice\\NumpyConv\\fc1_b.npy")
        self.fc2.weights = np.load("D:\\practice\\NumpyConv\\fc2_w.npy")
        self.fc2.bias = np.load("D:\\practice\\NumpyConv\\fc2_b.npy")
    
    def forward(self, data, train=False):
        net = self.conv1.forward(data)
        net = self.relu1.forward(net)
        net = self.pool1.forward(net)
        net = self.conv2.forward(net)
        net = self.relu2.forward(net)
        net = self.pool2.forward(net)
        net = self.fc1.forward(net)
        net = self.relu3.forward(net)
        net = self.dropout.forward(net, train=train)
        net = self.fc2.forward(net)
        return net
    
    def calLoss(self, labels, perdictions):
        loss = self.softmax.calLoss(labels, perdictions)
        return loss
    
    def gradient(self):
        error = self.softmax.gradient()
        error = self.fc2.gradient(error)
        error = self.dropout.gradient(error)
        error = self.relu3.gradient(error)
        error = self.fc1.gradient(error)
        error = self.pool2.gradient(error)
        error = self.relu2.gradient(error)
        error = self.conv2.gradient(error)
        error = self.pool1.gradient(error)
        error = self.relu1.gradient(error)
        error = self.conv1.gradient(error)

    def backward(self, learningRate=0.00001):
        self.gradient()
        self.conv1.backward(learningRate=learningRate)
        self.conv2.backward(learningRate=learningRate)
        self.fc1.backward(learningRate=learningRate)
        self.fc2.backward(learningRate=learningRate)

def main():
    mnistModel = Cmodel(1)
    temp = json.load(open("D:\\practice\\MnistTest\\MnistForm\\DrawDigit\\info.json"))
    data = np.array(temp[0]).reshape([1, 28, 28, 1])
    label = mnistModel.forward(data)
    print(np.argmax(label, 1))
    return np.argmax(label, 1)

if __name__ == "__main__":
    main()