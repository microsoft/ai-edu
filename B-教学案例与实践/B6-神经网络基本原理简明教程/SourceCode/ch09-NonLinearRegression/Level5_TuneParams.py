
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass2.NeuralNet2 import *
from HelperClass2.DataReader import *

x_data_name = "../../Data/ch09.train.npz"
y_data_name = "../../Data/ch09.test.npz"

def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

if __name__ == '__main__':
    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    n_input, n_hidden, n_output = 1, 2, 1
    eta, batch_size, max_epoch = 0.5, 10, 100000
    eps = 0.001

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet2(hp, "complex_121")

    net.train(dataReader, 50, True)
    net.ShowTrainingTrace()
    ShowResult(net, dataReader, hp.toString())




def train(ne, batch, eta):
    dataReader = DataReader(x_data_name, y_data_name)
    XData,YData = dataReader.ReadData()
    X = dataReader.NormalizeX(passthrough=True)
    Y = dataReader.NormalizeY()
    
    n_input, n_hidden, n_output = 1, ne, 1
    eta, batch_size, max_epoch = eta, batch, 10000
    eps = 0.001

    params = CParameters(n_input, n_hidden, n_output,
                         eta, max_epoch, batch_size, eps, 
                         LossFunctionName.MSE,
                         InitialMethod.Xavier)

    loss_history = CLossHistory(params)
    net = TwoLayerFittingNet()
    wb1, wb2 = net.train(dataReader, params, loss_history)
    return loss_history

def ShowLossHistory(file1, file2, file3, file4):
    lh = CLossHistory.Load(file1)
    axes = plt.subplot(2,2,1)
    lh.ShowLossHistory4(axes)
    
    lh = CLossHistory.Load(file2)
    axes = plt.subplot(2,2,2)
    lh.ShowLossHistory4(axes)

    lh = CLossHistory.Load(file3)
    axes = plt.subplot(2,2,3)
    lh.ShowLossHistory4(axes)

    lh = CLossHistory.Load(file4)
    axes = plt.subplot(2,2,4)
    lh.ShowLossHistory4(axes)

    plt.show()


def try_hyperParameters(ne, batch, eta):
    filename = str.format("{0}_{1}_{2}.pkl", ne, batch, eta).replace('.', '', 1)
    file = Path(filename)
    if file.exists():
        return file
    else:
        lh = train(ne, batch, eta)
        lh.Dump(filename)
        return file


if __name__ == '__main__':
  
    
    ne, batch, eta = 4, 10, 0.1
    file_1 = try_hyperParameters(ne, batch, eta)

    ne, batch, eta = 4, 10, 0.3
    file_2 = try_hyperParameters(ne, batch, eta)
    
    ne, batch, eta = 4, 10, 0.5
    file_3 = try_hyperParameters(ne, batch, eta)

    ne, batch, eta = 4, 10, 0.7
    file_4 = try_hyperParameters(ne, batch, eta)
    
    ShowLossHistory(file_1, file_2, file_3, file_4)
    
    ne, batch, eta = 4, 1, 0.5
    file_1 = try_hyperParameters(ne, batch, eta)

    ne, batch, eta = 4, 5, 0.5
    file_2 = try_hyperParameters(ne, batch, eta)

    # already have this data
    #ne, batch, eta = 4, 10, 0.5
    #lh = train(ne, batch, eta)
    #lh.Dump("4_10_05.pkl")

    ne, batch, eta = 4, 20, 0.5
    file_4 = try_hyperParameters(ne, batch, eta)
    
    ShowLossHistory(file_1, file_2, file_3, file_4)

    ne, batch, eta = 2, 10, 0.5
    file_1 = try_hyperParameters(ne, batch, eta)

    # already have this data
    #ne, batch, eta = 4, 10, 0.5
    #lh = train(ne, batch, eta)
    #lh.Dump("4_10_05.pkl")

    ne, batch, eta = 6, 10, 0.5
    file_3 = try_hyperParameters(ne, batch, eta)

    ne, batch, eta = 8, 10, 0.5
    file_4 = try_hyperParameters(ne, batch, eta)

    ShowLossHistory(file_1, file_2, file_3, file_4)


