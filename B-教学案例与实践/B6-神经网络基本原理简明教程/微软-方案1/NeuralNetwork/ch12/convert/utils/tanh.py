import numpy as np

class Ctanh(object):
    def __init__(self, inputSize, name=None, exname=""):
        self.shape = inputSize

        self.type = "Tanh"
        self.input_name = exname + "y"
        self.input_size = [1] + list(inputSize)[1:]
        self.output_name = name + "y"
        self.output_size = [1] + list(inputSize)[1:]