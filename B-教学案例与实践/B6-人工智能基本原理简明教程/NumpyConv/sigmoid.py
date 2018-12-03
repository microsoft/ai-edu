import numpy as np

class Csigmoid(object):
    def __init__(self, inputSize):
        self.shape = inputSize
    
    def forward(self, image):
        self.mask = 1 / (1 - np.exp(-1 * image))
        return self.mask

    def gradient(self, preError):
        self.mask = np.multiply(self.mask, 1 - self.mask)
        return np.multiply(preError, self.mask)

if __name__ == "__main__":
    img = np.ones([1, 3, 3, 1])
    relu = Csigmoid(img.shape)
    temp = relu.forward(img)
    temp = relu.gradient(temp)
    print(temp)