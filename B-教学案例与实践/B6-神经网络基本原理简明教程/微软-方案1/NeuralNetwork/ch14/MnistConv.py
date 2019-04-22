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
    return mdr

def Test(dataReader, model):
    correct = 0
    test_batch = 2
    max_iteration = dataReader.num_test//test_batch
    for i in range(max_iteration):
        x, y = dataReader.GetBatchTestSamples(test_batch, i)
        model.forward(x)
        result = np.argmax(model.output,axis=0).reshape(test_batch,-1)
        b = (result == y)
        correct += b.sum()

    return correct, dataReader.num_test

class Model(object):
    def __init__(self, param):
        self.c1 = ConvLayer((1,28,28), (4,5,5), (1,0), Sigmoid(), param)
        # 4x24x24
        self.p1 = PoolingLayer(self.c1.output_shape, (2,2,), 2, PoolingTypes.MAX)
        # 4x12x12
        self.f1 = FcLayer(self.p1.output_size, 32, Sigmoid(), param)
        self.f2 = FcLayer(self.f1.output_size, 10, Softmax(), param)

    def forward(self, x):
        a_c1 = self.c1.forward(x)
        a_p1 = self.p1.forward(a_c1)
        a_f1 = self.f1.forward(a_p1)
        a_f2 = self.f2.forward(a_f1)
        self.output = a_f2
        return self.output

    def backward(self, y):
        delta_in = self.output - y
        d_f2 = self.f2.backward(delta_in, LayerIndexFlags.LastLayer)
        d_f1 = self.f1.backward(d_f2, LayerIndexFlags.MiddleLayer)
        d_p1 = self.p1.backward(d_f1, LayerIndexFlags.MiddleLayer)
        d_c1 = self.c1.backward(d_p1, LayerIndexFlags.FirstLayer)

    def update(self, learning_rate):
        self.c1.update()
        self.f1.update()
        self.f2.update()

    def save(self):
        self.c1.save_parameters("c1")
        self.p1.save_parameters("p1")
        self.f1.save_parameters("f1")
        self.f2.save_parameters("f2")

    def load(self):
        self.c1.load_parameters("c1")
        self.p1.load_parameters("p1")
        self.f1.load_parameters("f1")
        self.f2.load_parameters("f2")


if __name__ == '__main__':

    num_output = 10
    dataReader = LoadData(num_output)
    max_epoch = 1
    batch_size = 1
    eta = 0.1
    eps = 0.01
    max_iteration = dataReader.num_example // batch_size
    params = CParameters(eta, max_epoch, batch_size, eps,
                    LossFunctionName.CrossEntropy3, 
                    InitialMethod.Xavier, 
                    OptimizerName.SGD)
    model = Model(params)
    
    model.load()
    print("testing...")
    c,n = Test(dataReader, model)
    print(str.format("rate={0} / {1} = {2}", c, n, c/n))
    
    exit()

    loss = 0 
    lossFunc = CLossFunction(LossFunctionName.CrossEntropy3)

    #list1 = []
    #list2 = []
    #list3 = []
    #list4 = []

    t0 = time.time()

    #max_iteration = 1000
    for epoch in range(max_epoch):
        print(epoch)
        for iteration in range(max_iteration):
            #t0 = time.time()
            batch_x, batch_y = dataReader.GetBatchSamples(batch_size, iteration)
            #plt.imshow(batch_x[0,0])
            #plt.show()
            #plt.imshow(batch_x[1,0])
            #plt.show()
            #t1 = time.time()
            model.forward(batch_x)
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
            if iteration % 100 == 0:
                print(iteration)
    
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
