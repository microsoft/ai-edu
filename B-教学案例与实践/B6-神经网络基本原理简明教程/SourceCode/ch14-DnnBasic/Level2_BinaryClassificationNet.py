# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

# x1=0,x2=0,y=0
# x1=0,x2=1,y=1
# x1=1,x2=0,y=1
# x1=1,x2=1,y=0
class XOR_DataReader(DataReader):
    def ReadData(self):
        self.X = np.array([0,0,1,1,0,1,0,1]).reshape(2,4)
        self.Y = np.array([0,1,1,0]).reshape(1,4)
        self.num_train = self.X.shape[1]
        self.num_feature = self.X.shape[0]

def LoadData():
    dataReader = XOR_DataReader(None, None)
    dataReader.ReadData()
    return dataReader

def ShowResult(net, title):
    print("waiting for 10 seconds...")
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(2,1)
            output = net.inference(x)
            if output[0,0] >= 0.5:
                plt.plot(x[0,0], x[1,0], 's', c='m')
            else:
                plt.plot(x[0,0], x[1,0], 's', c='y')
            # end if
        # end for
    # end for
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    dataReader = LoadData()
    num_input = dataReader.num_feature
    num_hidden1 = 2
    num_output = 1

    max_epoch = 10000
    batch_size = 1
    learning_rate = 0.1
    eps = 0.01

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy2, 
                        InitialMethod.Xavier, 
                        OptimizerName.SGD)

    net = NeuralNet(params)
    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivatorLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    fc2 = FcLayer(num_hidden1, num_output, params)
    net.add_layer(fc2, "fc2")
    sigmoid2 = ClassificationLayer(Sigmoid())
    net.add_layer(sigmoid2, "sigmoid2")

    net.train(dataReader, checkpoint=10, need_test=False)
    net.ShowLossHistory()
    
    ShowResult(net, params.toString())
