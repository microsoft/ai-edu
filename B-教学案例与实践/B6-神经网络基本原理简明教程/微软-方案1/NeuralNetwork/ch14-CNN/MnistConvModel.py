# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

# coding: utf-8

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet import *
from MiniFramework.GDOptimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.Activators import *
from MiniFramework.ConvLayer import *
from MiniFramework.PoolingLayer import *

from MnistImageReader import *

train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def LoadData(num_output):
    mdr = MnistImageReader(train_image_file, train_label_file, test_image_file, test_label_file)
    mdr.ReadData()
    mdr.Normalize()
    mdr.Shuffle()
    mdr.GenerateDevSet(12)
    return mdr

def Test(dataReader, model):
    correct = 0
    test_batch = 1000
    max_iteration = dataReader.num_test//test_batch
    for i in range(max_iteration):
        x, y = dataReader.GetBatchTestSamples(test_batch, i)
        model.forward(x)
        correct += CalAccuracy(model.output, None, y)
    #end for
    return correct, dataReader.num_test

def CheckErrorAndLoss(dataReader, model, lossFunc, batch_x, batch_y, loss_history, epoch, iteration):
    
    loss_train = lossFunc.CheckLoss(batch_y, model.output)
    accuracy_train = CalAccuracy(model.output, batch_y, None) / batch_y.shape[1]
    
    model.forward(dataReader.XDevSet)
    loss_val = lossFunc.CheckLoss(dataReader.YDevSet.T, model.output)
    y = dataReader.YDevSet.T
    accuracy_val = CalAccuracy(model.output, y, None) / y.shape[1]
    
    loss_history.Add(epoch, iteration, loss_train, accuracy_train, loss_val, accuracy_val)
    print("epoch=%d, iteration=%d" %(epoch, iteration))
    print("loss_train=%.3f, accuracy_train=%f" %(loss_train, accuracy_train))
    print("loss_valid=%.3f, accuracy_valid=%f" %(loss_val, accuracy_val))

    #val_batch = 5
    #max_iteration = dataReader.num_validation//val_batch
    #for i in range(max_iteration):
        #x, y = dataReader.GetBatchValidationSamples(val_batch, i)
        #model.forward(x)
        #correct += CalAccuracy(model.output, y, None)
    #end for
    return

def CalAccuracy(a, y_onehot, y_label):
    ra = np.argmax(a, axis=0).reshape(-1,1)
    if y_onehot is None:
        ry = y_label
    elif y_label is None:
        ry = np.argmax(y_onehot, axis=0).reshape(-1,1)
    r = (ra == ry)
    correct = r.sum()
    return correct

class Model(object):
    def __init__(self, param):
        self.c1 = ConvLayer((1,28,28), (4,3,3), (2,2), Relu(), param)
        # 4x24x24
        self.p1 = PoolingLayer(self.c1.output_shape, (2,2,), 2, PoolingTypes.MAX)
        # 4x12x12
        #self.c2 = ConvLayer(self.p1.output_shape, (8,3,3), (1,0), Relu(), param)
        # 4x10x10
        #self.p2 = PoolingLayer(self.c2.output_shape, (2,2,), 2, PoolingTypes.MAX)
        # 4x5x5
        #self.f1 = FcLayer(self.p2.output_size, 32, Relu(), param)
        self.f1 = FcLayer(self.p1.output_size, 32, Relu(), param)
        self.f2 = FcLayer(self.f1.output_size, 10, Softmax(), param)

    def forward(self, x):
        net = self.c1.forward(x)
        net = self.p1.forward(net)
        #net = self.c2.forward(net)
        #net = self.p2.forward(net)
        net = self.f1.forward(net)
        net = self.f2.forward(net)
        self.output = net
        return self.output

    def backward(self, y):
        delta = self.output - y
        delta = self.f2.backward(delta, LayerIndexFlags.LastLayer)
        delta = self.f1.backward(delta, LayerIndexFlags.MiddleLayer)
        #delta = self.p2.backward(delta, LayerIndexFlags.MiddleLayer)
        #delta = self.c2.backward(delta, LayerIndexFlags.MiddleLayer)
        delta = self.p1.backward(delta, LayerIndexFlags.MiddleLayer)
        delta = self.c1.backward(delta, LayerIndexFlags.FirstLayer)

    def update(self):
        self.c1.update()
        #self.c2.update()
        self.f1.update()
        self.f2.update()

    def save(self):
        self.c1.save_parameters("c1")
        self.p1.save_parameters("p1")
        #self.c2.save_parameters("c2")
        #self.p2.save_parameters("p2")
        self.f1.save_parameters("f1")
        self.f2.save_parameters("f2")

    def load(self):
        self.c1.load_parameters("c1")
        self.p1.load_parameters("p1")
        #self.c2.load_parameters("c2")
        #self.p2.load_parameters("p2")
        self.f1.load_parameters("f1")
        self.f2.load_parameters("f2")

def train():

    num_output = 10
    dataReader = LoadData(num_output)

    max_epoch = 1
    batch_size = 50
    eta = 0.01
    eps = 0.01
    max_iteration = dataReader.num_train // batch_size
    params = CParameters(eta, max_epoch, batch_size, eps,
                    LossFunctionName.CrossEntropy3, 
                    InitialMethod.Xavier, 
                    OptimizerName.SGD)
    model = Model(params)
    """
    model.load()
    
    print("testing...")
    c,n = Test(dataReader, model)
    print(str.format("rate={0} / {1} = {2}", c, n, c/n))
    exit()
    """
    loss = 0 
    lossFunc = CLossFunction(LossFunctionName.CrossEntropy3)
    loss_history = CLossHistory()


    t0 = time.time()

    #max_iteration = 1000
    for epoch in range(max_epoch):
        print(epoch)
        for iteration in range(max_iteration):
            #t0 = time.time()
            batch_x, batch_y = dataReader.GetBatchTrainSamples(batch_size, iteration)
            output = model.forward(batch_x)
            model.backward(batch_y)
            model.update()

            # calculate loss
            if iteration % 100 == 0:
                CheckErrorAndLoss(dataReader, model, lossFunc, batch_x, batch_y, loss_history, epoch, iteration*batch_size)
            # end if
        # end for
        model.save()
        dataReader.Shuffle()
        
    
    t1 = time.time()
    print("time used:", t1 - t0)

    print("testing...")
    c,n = Test(dataReader, model)
    print(str.format("rate={0} / {1} = {2}", c, n, c / n))

    model.save()
    loss_history.ShowLossHistory(params)

if __name__ == '__main__':
    train()