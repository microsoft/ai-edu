import numpy as np

class Cpool(object):
    def __init__(self, inputSize, kernelSize=2, stride=2, name="", exname=""):
        self.shape = inputSize
        self.kernelSize = kernelSize
        self.stride = stride
        self.outputChannels = inputSize[-1]
        self.batchSize = inputSize[0]
        len = (self.shape[1] - kernelSize) // stride + 1
        self.output = np.zeros((self.batchSize, len, len, self.outputChannels))
        self.outputShape = self.output.shape

        self.type = "MaxPool"
        self.input_name = f'{exname}y' if exname else f'{name}x'
        self.input_size = [1] + list(inputSize)[1:]
        self.output_name = name + "y"
        self.output_size = [1] + list(self.outputShape)[1:]
        self.kernel = kernelSize
        self.pad = 0
    
    def forward(self, image):
        self.memory = np.zeros(image.shape)

        for b in range(self.batchSize):
            for c in range(self.outputChannels):
                for i in range(0, image.shape[1], self.stride):
                    for j in range(0, image.shape[2], self.stride):
                        self.output[b, i // self.stride, j // self.stride, c] = np.max(
                            image[b, i:i + self.kernelSize, j:j + self.kernelSize, c]
                        )
                        index = np.argmax(image[b, i:i + self.kernelSize, j:j+ self.kernelSize, c])
                        self.memory[b, i + index % self.stride, j + index % self.stride, c] = 1
        return self.output

    def gradient(self, preError):
        preError = np.repeat(preError, self.stride, axis=1)
        preError = np.repeat(preError, self.stride, axis=2)
        return np.multiply(self.memory, preError)