import numpy as np
from ONNXConverter.conv2d import Cconv2d
from ONNXConverter.pool import Cpool
from ONNXConverter.relu import Crelu
from ONNXConverter.softmax import Csoftmax
from ONNXConverter.sigmoid import Csigmoid
from ONNXConverter.tanh import Ctanh
from ONNXConverter.fc import Cfc
from ONNXConverter.dropout import Cdropout
from ONNXConverter.save import model_save
from ONNXConverter.transfer import ModelTransfer


# some configuration
# these variables records the path for the strored weights
fc1_w_path = "./Level3_w1.npy"
fc1_b_path = "./Level3_b1.npy"
fc2_w_path = "./Level3_w2.npy"
fc2_b_path = "./Level3_b2.npy"
fc3_w_path = "./Level3_w3.npy"
fc3_b_path = "./Level3_b3.npy"

fc1_w = np.load(fc1_w_path)
fc1_b = np.load(fc1_b_path)
fc2_w = np.load(fc2_w_path)
fc2_b = np.load(fc2_b_path)
fc3_w = np.load(fc3_w_path)
fc3_b = np.load(fc3_b_path)

outputshape1 = fc1_w.shape[0]
outputshape2 = fc2_w.shape[0]
outputshape3 = fc3_w.shape[0]

# define the model structure
class Cmodel(object):
    def __init__(self):
        """The Model definition."""
        self.fc1 = Cfc([1,784], outputshape1, name="fc1", exname="")
        self.activation1 = Csigmoid(self.fc1.outputShape, name="activation1", exname="fc1")
        self.fc2 = Cfc(self.fc1.outputShape, outputshape2, name="fc2", exname="activation1")
        self.activation2 = Ctanh(self.fc1.outputShape, name="activation2", exname="fc2")
        self.fc3 = Cfc(self.fc2.outputShape, outputshape3, name="fc3", exname="activation2")
        self.activation3 = Csoftmax([1, outputshape3], name="activation3", exname="fc3")

        self.model = [
          self.fc1, self.activation1, self.fc2, self.activation2, self.fc3, self.activation3,  
        ]

        self.fc1.weights = fc1_w.T
        self.fc1.bias = fc1_b.reshape(self.fc1.bias.shape)
        self.fc2.weights = fc2_w.T
        self.fc2.bias = fc2_b.reshape(self.fc2.bias.shape)
        self.fc3.weights = fc3_w.T
        self.fc3.bias = fc3_b.reshape(self.fc3.bias.shape)

    def save_model(self, path="./"):
        
        model_path = path + "model.json" 
        model_save(self.model, path) 
        ModelTransfer(model_path, path + "my_mnist.onnx")

if __name__ == '__main__':

    # attention: need ONNX 1.4.1 python package
    # pip install --upgrade onnx

    model = Cmodel()
    model.save_model()
    print("Succeed! Your model file is <my_mnist.onnx>")

