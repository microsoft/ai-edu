
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from HelperClass2.NeuralNet_2_0 import *

train_data_name = "../../Data/ch09.train.npz"
test_data_name = "../../Data/ch09.test.npz"

def train(hp, folder):
    net = NeuralNet_2_0(hp, folder)
    net.train(dataReader, 50, True)
    trace = net.GetTrainingHistory()
    return trace


def ShowLossHistory(folder, file1, hp1, file2, hp2, file3, hp3, file4, hp4):
    lh = TrainingHistory_2_0.Load(file1)
    axes = plt.subplot(2,2,1)
    lh.ShowLossHistory4(axes, hp1)
    
    lh = TrainingHistory_2_0.Load(file2)
    axes = plt.subplot(2,2,2)
    lh.ShowLossHistory4(axes, hp2)

    lh = TrainingHistory_2_0.Load(file3)
    axes = plt.subplot(2,2,3)
    lh.ShowLossHistory4(axes, hp3)

    lh = TrainingHistory_2_0.Load(file4)
    axes = plt.subplot(2,2,4)
    lh.ShowLossHistory4(axes, hp4)

    plt.show()


def try_hyperParameters(folder, n_hidden, batch_size, eta):
    hp = HyperParameters_2_0(1, n_hidden, 1, eta, 10000, batch_size, 0.001, NetType.Fitting, InitialMethod.Xavier)
    filename = str.format("{0}\\{1}_{2}_{3}.pkl", folder, n_hidden, batch_size, eta).replace('.', '', 1)
    file = Path(filename)
    if file.exists():
        return file, hp
    else:
        lh = train(hp, folder)
        lh.Dump(file)
        return file, hp


if __name__ == '__main__':
  
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()
    
    folder = "complex_turn"

    ne, batch, eta = 4, 10, 0.1
    file_1, hp1 = try_hyperParameters(folder, ne, batch, eta)

    ne, batch, eta = 4, 10, 0.3
    file_2, hp2 = try_hyperParameters(folder, ne, batch, eta)
    
    ne, batch, eta = 4, 10, 0.5
    file_3, hp3 = try_hyperParameters(folder, ne, batch, eta)

    ne, batch, eta = 4, 10, 0.7
    file_4, hp4 = try_hyperParameters(folder, ne, batch, eta)
    
    ShowLossHistory(folder, file_1, hp1, file_2, hp2, file_3, hp3, file_4, hp4)
    
    ne, batch, eta = 4, 5, 0.5
    file_1, hp1 = try_hyperParameters(folder, ne, batch, eta)

    ne, batch, eta = 4, 10, 0.5
    file_2, hp2 = try_hyperParameters(folder, ne, batch, eta)

    # already have this data
    ne, batch, eta = 4, 15, 0.5
    file_3, hp3 = try_hyperParameters(folder, ne, batch, eta)

    ne, batch, eta = 4, 20, 0.5
    file_4, hp4 = try_hyperParameters(folder, ne, batch, eta)
    
    ShowLossHistory(folder, file_1, hp1, file_2, hp2, file_3, hp3, file_4, hp4)

    ne, batch, eta = 2, 10, 0.5
    file_1, hp1 = try_hyperParameters(folder, ne, batch, eta)

    ne, batch, eta = 4, 10, 0.5
    file_2, hp2 = try_hyperParameters(folder, ne, batch, eta)

    ne, batch, eta = 6, 10, 0.5
    file_3, hp3 = try_hyperParameters(folder, ne, batch, eta)

    ne, batch, eta = 8, 10, 0.5
    file_4, hp4 = try_hyperParameters(folder, ne, batch, eta)

    ShowLossHistory(folder, file_1, hp1, file_2, hp2, file_3, hp3, file_4, hp4)


