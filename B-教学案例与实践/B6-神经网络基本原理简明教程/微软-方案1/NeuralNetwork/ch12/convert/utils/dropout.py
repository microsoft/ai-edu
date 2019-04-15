import numpy as np

class Cdropout(object):
    def __init__(self, inputSize, prob):
        self.outputShape = inputSize
        self.prob = prob
    
    def forward(self, data, train=False):
        self.data = data
        self.train = train
        self.mask = np.random.rand(
            self.outputShape[0], self.outputShape[1]) > self.prob
        if train:
            return np.multiply(data, self.mask) / (1 - self.prob)
        else:
            return data
    
    def gradient(self, preError):
        if self.train:
            return np.multiply(preError, self.mask)
        else:
            return preError