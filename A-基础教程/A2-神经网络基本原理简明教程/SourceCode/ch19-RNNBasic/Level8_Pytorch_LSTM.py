# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


import numpy as np
import matplotlib.pyplot as plt
from MiniFramework.EnumDef_6_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.LossFunction_1_1 import *
from MiniFramework.TrainingHistory_3_0 import *
from MiniFramework.HyperParameters_4_3 import *
from MiniFramework.WeightsBias_2_1 import *
from ExtendedDataReader.NameDataReader import *
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader


file = "../../data/ch19.name_language.txt"

def load_data():
    dr = NameDataReader()
    dr.ReadData(file)
    dr.GenerateValidationSet(1000)
    return dr

def process_data(dr):               # 统一输入单词的长度，不足补0，此方法会增加训练时间
    """

    :param dr: class data reader
    :return: process X, Y
    """
    new_X = []
    new_Y = []
    for i in range(len(dr.X)):
        for j in range(dr.X[i].shape[0]):
            new_X.append(np.pad(dr.X[i][j], ((0, 17-dr.X[i].shape[1]), (0, 0)), 'constant', constant_values=(0, 0)))
    for i in range(len(dr.Y)):
        for j in range(dr.Y[i].shape[0]):
            new_Y.append(dr.Y[i][j])
    new_Y = np.argmax(new_Y, axis=1)            # new_Y (4400,10) ----> (4400,)  ont-hot编码改成Label encoding
    return np.array(new_X), np.array(new_Y)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=29,          # character num.
            hidden_size=8,         # RNN or LSTM hidden layer, 设置的稍大一些可能效果更佳，此处仅作对比
            num_layers=1,
            batch_first=True,
            bidirectional=True      # 双向LSTM, 若设置为False,对应hidden_size增大两倍
        )
        self.out = nn.Linear(16, 10)
        self.softmax = nn.Softmax()         # classification, softmax

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        out = self.softmax(out)
        return out

if __name__ == '__main__':
    max_epoch = 100      # hyper-parameters
    lr = 0.2
    rnn = RNN()

    # Data processing
    dataReader = load_data()
    X, Y = process_data(dataReader)         # X ---  (4400, 17, 29)
    tensor_X, tensor_Y = torch.FloatTensor(X), torch.LongTensor(Y)      # transform numpy to tensor
    torch_dataset = TensorDataset(tensor_X, tensor_Y)
    train_loader = DataLoader(              # data loader class
        dataset=torch_dataset,
        batch_size=8,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()       # classification  --- CrossEntropyLoss

    plot_y_l = []               # record loss
    plot_y_a = []               # record accuracy
    for epoch in range(max_epoch):
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = rnn(batch_X)
            loss = loss_func(pred, batch_Y)
            loss.backward()
            optimizer.step()
            pred = pred.data.numpy()
            pred = np.argmax(pred, axis=1)

        pred = rnn(tensor_X)
        loss = loss_func(pred, tensor_Y)
        pred = pred.data.numpy()
        pred = np.argmax(pred, axis=1)
        accuracy = np.sum(Y == pred) / np.shape(pred)[0]
        plot_y_l.append(float(loss.data.numpy()))
        plot_y_a.append(accuracy)
        print("[Epoch:%d], Loss: %.5f, Accuracy: %.5f" % (epoch, loss.data.numpy(),accuracy))
    plt.subplot(121)
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.plot([i for i in range(max_epoch)], plot_y_l)
    plt.subplot(122)
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    plt.plot([i for i in range(max_epoch)], plot_y_a)
    plt.show()
