
from ExtendedDataReader.MnistImageDataReader import *

train_image_file = os.path.join(os.path.dirname(__file__), 'data', 'fashion-train-images-idx3-ubyte')
train_label_file = os.path.join(os.path.dirname(__file__), 'data', 'fashion-train-labels-idx1-ubyte')
test_image_file = os.path.join(os.path.dirname(__file__), 'data', 'fashion-t10k-images-idx3-ubyte')
test_label_file = os.path.join(os.path.dirname(__file__), 'data', 'fashion-t10k-labels-idx1-ubyte')

class FashionMnistDataReader(MnistImageDataReader):
    # mode: "image"=Nx1x28x28,  "vector"=1x784
    def __init__(self, mode="image"):
        self.train_image_file = train_image_file
        self.train_label_file = train_label_file
        self.test_image_file = test_image_file
        self.test_label_file = test_label_file
        self.num_example = 0
        self.num_feature = 0
        self.num_category = 0
        self.num_validation = 0
        self.num_test = 0
        self.num_train = 0
        self.mode = mode    # image or vector
