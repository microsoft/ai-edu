# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


import numpy as np
import matplotlib.pyplot as plt
from MiniFramework.EnumDef_6_0 import *
from MiniFramework.DataReader_2_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.LossFunction_1_1 import *
from MiniFramework.TrainingHistory_3_0 import *
from MiniFramework.LSTMCell_1_2 import *
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.init import xavier_normal


train_file = "../../data/ch19.train_minus.npz"
test_file = "../../data/ch19.test_minus.npz"

def load_data():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.Shuffle()
    dr.GenerateValidationSet(k=0)
    return dr

def process_data(dr):               # 统一输入单词的长度，不足补0，此方法会增加训练时间
    """
    :param dr: class data reader
    :return: process X, Y
    """

    return dr.XTrain, dr.XTest, dr.YTrain, dr.YTest             # 这是为了统一一个表达方式，和之前的章节一样

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_normal(m.weight.data)
        xavier_normal(m.bias.data)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=2,          # character num.
            hidden_size=4,         # RNN or LSTM hidden layer, 设置的稍大一些可能效果更佳，此处仅作对比
            num_layers=1,
            batch_first=True,
            bidirectional=True,      # 双向LSTM, 若设置为False,对应hidden_size增大两倍

        )
        self.softmax = nn.Softmax()         # classification, softmax
        self.fc = nn.Linear(8, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)           # 多对多的序列            (2, 4, 8)
        out = self.fc(r_out)                            # 0和1是二分类问题       （2, 4, 2)
        out = self.sigmoid(out)
        return out

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.rnn = nn.LSTM(
            input_size=2,
            hidden_size=4,
            num_layers=1,
            batch_first=True,
            bidirectional=True,      # 双向GRU, 若设置为False,对应hidden_size增大两倍

        )
        self.softmax = nn.Softmax()         # classification, softmax
        self.fc = nn.Linear(8, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)           # 多对多的序列            (2, 4, 8)
        out = self.fc(r_out)                            # 0和1是二分类问题       （2, 4, 2)
        out = self.sigmoid(out)
        return out

def accracy_score(pred, label):
    """

    :param pred:  prediction
    :param label:   real_label
    :return:    Accuracy score
    """
    pred = np.argmax(pred, axis=1)                      # argmax取出预测值 0, 1
    label_t = label.reshape(4 * label.shape[0],)        # (136, 4) ----> (544,)

    # 每四行全对就代表预测正确了一个序列
    tmp = [sum(np.array(pred[4 * i:4 * (i+1)]) == np.array(label_t[4*i:4*(i+1)])) for i in range(len(pred)//4)]
    acc_num = [i == 4 for i in tmp]
    accuracy = sum(acc_num)/label.shape[0]
    return accuracy


if __name__ == '__main__':
    max_epoch = 100      # hyper-parameters
    lr = 1e-2
    batch_size = 2
    # rnn = LSTM()        # LSTM  model
    rnn = GRU()         # GRU model
    rnn.apply(weights_init)

    # Data processing
    dataReader = load_data()
    X, XTest, Y, YTest = process_data(dataReader)      # X ---  (136, 4, 2)
    tensor_X, tensor_Y = torch.FloatTensor(XTest), torch.LongTensor(YTest)      # transform numpy to tensor
    torch_dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(Y))
    train_loader = DataLoader(              # data loader class
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()       # classification  --- CrossEntropyLoss

    plot_y_l = []               # record loss
    plot_y_a = []               # record accuracy
    for epoch in range(max_epoch):
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            num_data = batch_X.size(0)
            batch_Y = batch_Y.view(batch_size*4, )             # (batch_size, 4) ---> (4*batch_size, ) 序列二分类
            optimizer.zero_grad()
            pred = rnn(batch_X).view(batch_size*4,2)
            loss = loss_func(pred, batch_Y)

            # loss = loss_func(pred, batch_Y)
            loss.backward()
            optimizer.step()
            pred = pred.data.numpy()
            pred = np.argmax(pred, axis=1)

        pred = rnn(tensor_X).view(tensor_X.size(0)*4, 2)
        tensor_Y = tensor_Y.view(tensor_X.size(0)*4, )
        loss = loss_func(pred, tensor_Y)
        pred = pred.data.numpy()
        # print(pred)
        accuracy = accracy_score(pred, YTest)
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
