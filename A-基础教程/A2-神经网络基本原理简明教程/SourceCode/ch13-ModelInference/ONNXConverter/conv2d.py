import numpy as np
import sys
import math
from functools import reduce

class Cconv2d(object):
    def __init__(self, inputSize, kernelSize, outputChannel, stride=1, padding="SAME", name="", exname=""):
        self.inputSize = inputSize
        self.kernelSize = kernelSize
        self.stride = stride
        self.batch = self.inputSize[0]
        self.inputChannel = inputSize[-1]
        self.outputChannel = outputChannel
        self.padding = padding
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, inputSize)) / outputChannel
        self.weights = np.random.standard_normal((kernelSize, kernelSize, self.inputChannel, outputChannel)) / weights_scale
        self.bias = np.random.standard_normal(outputChannel) / weights_scale
        self.weightsGrad = np.zeros(self.weights.shape)
        self.biasGrad = np.zeros(self.bias.shape)
        self.backError = np.zeros((inputSize[0], inputSize[1] // stride, inputSize[2] // stride, outputChannel))
        self.outputShape = self.backError.shape
        
        self.type = "Conv"
        
        self.input_name = f'{exname}y' if exname else f'{name}x'
        self.output_name = name + "y"
        self.input_size = [1] + list(inputSize)[1:]
        self.output_size = [1] + list(self.backError.shape)[1:]

        self.kernel = kernelSize
        self.stride = stride
        self.pad = kernelSize // 2
        
        self.weights_name = name + "w"
        self.bias_name = name + "b"
        self.weights_size = self.weights.shape
        self.bias_size = self.bias.shape


    
    def expand(self, image, kernelSize, stride):
        colImage = []
        for i in range(0, image.shape[1] - kernelSize + 1, stride):
            for j in range(0, image.shape[2] - kernelSize + 1, stride):
                col = image[:, i:i + kernelSize, j:j+kernelSize, :].reshape(-1)
                colImage.append(col)
        return np.array(colImage)


    def forward(self, image):
        weights = np.reshape(self.weights, [-1, self.outputChannel])
        self.image = image
        shape = image.shape
        image = np.pad(image, 
        ((0, 0), (self.kernelSize // 2, self.kernelSize // 2), (self.kernelSize // 2, self.kernelSize // 2), (0, 0)),
        mode='constant', constant_values=0)
        
        self.colImage = []
        convOut = np.zeros(self.backError.shape)
        for i in range(shape[0]):
            colImage = self.expand(image[i][np.newaxis, :], self.kernelSize, self.stride)
            convOut[i] = np.reshape(np.dot(colImage, weights) + self.bias, self.backError[0].shape)
            self.colImage.append(colImage)
        
        self.colImage = np.array(self.colImage)
        
        return convOut

    def gradient(self, preError):
        self.backError = preError
        preError = np.reshape(preError, [self.inputSize[0], -1, self.outputChannel])
        
        for i in range(self.inputSize[0]):
            self.weightsGrad = self.weightsGrad + np.dot(self.colImage[i].T, preError[i]).reshape(self.weights.shape)
        self.biasGrad = self.biasGrad + np.sum(preError, (0, 1))
        
        preError = np.pad(self.backError, 
        ((0, 0), (self.kernelSize // 2, self.kernelSize //2), (self.kernelSize // 2, self.kernelSize // 2), (0, 0)),
        mode='constant', constant_values=0)
        weights = np.flipud(np.fliplr(self.weights))
        weights = weights.swapaxes(2, 3)
        weights = np.reshape(weights, [-1, self.inputChannel])
        backError = np.zeros(self.inputSize)
        for i in range(self.inputSize[0]):
            backError[i] = np.reshape(np.dot(self.expand(preError[i][np.newaxis, :], self.kernelSize, self.stride), weights), self.inputSize[1:4])
        
        return backError

    def backward(self, learningRate=0.0001, weightsDecay=0.0004):
        weights = (1 - weightsDecay) * self.weights
        bias = (1 - weightsDecay) * self.bias
        self.weights = weights - learningRate * self.weightsGrad
        self.bias = bias - learningRate * self.biasGrad

        self.weightsGrad = 0.9 * self.weightsGrad
        self.biasGrad = 0.9 * self.biasGrad


if __name__ == "__main__":
    img = np.ones((1, 1, 1, 3))
    img *= 2
    conv = Cconv2d(img.shape, 1, 1)
    temp = conv.forward(img)
    conv.gradient(np.ones((1, 1, 1, 1)))
    # print(conv.weightsGrad)
    # print(conv.biasGrad)
    # print(conv.weights)