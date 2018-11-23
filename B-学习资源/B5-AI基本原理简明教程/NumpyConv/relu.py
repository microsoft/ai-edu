import numpy as np

class Crelu(object):
    def __init__(self, inputSize):
        self.shape = inputSize
    
    def forward(self, image):
        self.memory = np.zeros(self.shape)
        self.memory[image > 0] = 1
        return np.maximum(image, 0)

    def gradient(self, preError):
        return np.multiply(preError, self.memory)

if __name__ == "__main__":
    img = np.ones([1, 3, 3, 1])
    relu = Crelu(img.shape)
    temp = relu.forward(img)
    temp = relu.gradient(temp)
    print(temp)