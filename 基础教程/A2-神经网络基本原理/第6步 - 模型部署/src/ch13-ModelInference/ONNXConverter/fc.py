import numpy as np
import sys
import math
from functools import reduce

class Cfc(object):
    def __init__(self, inputSize, outputNum=2, name=None, exname=""):
        self.shape = inputSize
        self.batch = inputSize[0]
        self.weights = np.random.standard_normal((reduce(lambda x, y: x * y, self.shape[1:]), outputNum)) / 100
        self.bias = np.random.standard_normal(outputNum) / 100
        self.output = np.zeros((self.batch, outputNum))
        self.weightsGrad = np.zeros(self.weights.shape)
        self.biasGrad = np.zeros(self.bias.shape)
        self.outputShape = self.output.shape

        self.type = "FC"
        
        self.input_name = f'{exname}y' if exname else f'{name}x'
        self.output_name = name + "y"
        self.input_size = [1] + list(inputSize)[1:]
        self.output_size = [1] + list(self.outputShape)[1:]
        
        self.weights_name = name + "w"
        self.bias_name = name + "b"
        self.weights_size = self.weights.shape
        self.bias_size = self.bias.shape
    
    def forward(self, image):
        image = np.reshape(image, [self.batch, -1])
        fcout = np.dot(image, self.weights) + self.bias
        self.output = fcout
        self.image = image
        return fcout
    
    def gradient(self, preError):
        for i in range(self.batch):
            imagei = self.image[i][:, np.newaxis]
            preErrori = preError[i][:, np.newaxis].T
            self.weightsGrad = self.weightsGrad + np.dot(imagei, preErrori)
            self.biasGrad = self.biasGrad + np.reshape(preErrori, self.biasGrad.shape)
        
        return np.reshape(np.dot(preError, self.weights.T), self.shape)
    
    def backward(self, learningRate=0.001, weightsDecay=0.004):
        weights = (1 - weightsDecay) * self.weights
        bias = (1 - weightsDecay) * self.bias
        self.weights = weights - learningRate * self.weightsGrad
        self.bias = bias - learningRate * self.biasGrad
        self.weightsGrad = np.zeros(self.weights.shape)
        self.biasGrad = np.zeros(self.bias.shape)

if __name__ == "__main__":
    img = np.ones((2, 28, 28, 3))
    img *= 2
    fc = Cfc(img.shape)
    temp = fc.forward(img)
    temp1 = temp + 1
    fc.gradient(temp1 - temp)
    print(fc.weightsGrad)
    print(fc.biasGrad)