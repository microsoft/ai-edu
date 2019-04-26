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
    mdr.GenerateDevSet()
    return mdr

def Test(dataReader, model):
    correct = 0
    test_batch = 5
    max_iteration = dataReader.num_test//test_batch
    for i in range(max_iteration):
        x, y = dataReader.GetBatchTestSamples(test_batch, i)
        model.forward(x)
        correct += CalAccuracy(model.output, None, y)
    #end for
    return correct, dataReader.num_test

def Validate(dataReader, model):
    correct = 0
    val_batch = 5
    max_iteration = dataReader.num_validation//val_batch
    for i in range(max_iteration):
        x, y = dataReader.GetBatchValidationSamples(val_batch, i)
        model.forward(x)
        correct += CalAccuracy(model.output, y, None)
    #end for
    return correct, dataReader.num_validation

def CalAccuracy(a, y_onehot, y_label):
    ra = np.argmax(a, axis=0)
    if y_onehot is None:
        ry = y_label
    elif y_label is None:
        ry = np.argmax(y_onehot, axis=0)
    r = (ra == ry)
    correct = r.sum()
    return correct

class Model(object):
    def __init__(self, param):
        self.c1 = ConvLayer((1,28,28), (4,5,5), (1,0), Relu(), param)
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

    def update(self, learning_rate):
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
    batch_size = 5
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

    #list1 = []
    #list2 = []
    #list3 = []
    #list4 = []

    t0 = time.time()

    #max_iteration = 1000
    for epoch in range(max_epoch):
        print(epoch)
        #dataReader.Shuffle()
        for iteration in range(max_iteration):
            #t0 = time.time()
            batch_x, batch_y = dataReader.GetBatchTrainSamples(batch_size, iteration)
            #plt.imshow(batch_x[0,0])
            #plt.show()
            #plt.imshow(batch_x[1,0])
            #plt.show()
            #t1 = time.time()
            output = model.forward(batch_x)
            #t2 = time.time()
            model.backward(batch_y)
            #t3 = time.time()
            model.update(eta)
            #t4 = time.time()
            """
            list1.append(t1-t0)
            list2.append(t2-t1)
            list3.append(t3-t2)
            list4.append(t4-t3)
            """
            # calculate loss
            if iteration % 100 == 0:
                loss = lossFunc.CheckLoss(batch_y, model.output)
                accuracy = CalAccuracy(model.output, batch_y, None)
                loss_history.AddLossHistory(loss, epoch, iteration, accuracy)
                print("epoch=%d, iteration=%d, loss=%f, accuracy=%f" %(epoch, iteration * batch_size, loss, accuracy))
            # end if
        # end for
        model.save()
        c,n = Validate(dataReader, model)
        print(str.format("rate={0} / {1} = {2}", c, n, c / n))
        dataReader.Shuffle()
        dataReader.GenerateDevSet()
        
    
    t1 = time.time()
    print("time used:", t1 - t0)
    

    """        
    t1,t2,t3,t4=0,0,0,0
    for i in range(1000):
        t1 += list1[i]
        t2 += list2[i]
        t3 += list3[i]
        t4 += list4[i]
    print(t1,t2,t3,t4)
    """

    print("testing...")
    c,n = Test(dataReader, model)
    print(str.format("rate={0} / {1} = {2}", c, n, c / n))



    """
    model.forward(dataReader.X.reshape(60000,1,28,28))
    loss = lossFunc.CheckLoss(dataReader.Y, model.output)
    print("epoch=%d, iteration=%d, loss=%f" %(epoch,iteration,loss))
    is_min = loss_history.AddLossHistory(loss, epoch, iteration)                
    """

    model.save()
    loss_history.ShowLossHistory(params)

if __name__ == '__main__':
    train()