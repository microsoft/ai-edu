from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import torch.nn.init as init
import warnings
warnings.filterwarnings('ignore')

train_file = "../../Data/ch14.Income.train.npz"
test_file = "../../Data/ch14.Income.test.npz"

def LoadData():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

class Model(nn.Module):         # add bach norm
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(14, 32, bias=True)
        self.bn1 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(32, 64, bias=True)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32, bias=True)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 16, bias=True)
        self.bn4 = nn.BatchNorm1d(16)

        self.fc5 = nn.Linear(16, 8, bias=True)
        self.bn5 = nn.BatchNorm1d(8)

        self.fc6 = nn.Linear(8, 4, bias=True)
        self.bn6 = nn.BatchNorm1d(4)

        self.fc7 = nn.Linear(4, 2, bias=True)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.bn4(x)
        x = F.leaky_relu(self.fc5(x))
        x = self.bn5(x)
        x = F.leaky_relu(self.fc6(x))
        x = self.bn6(x)
        x = F.sigmoid(self.fc7(x))
        return x

    def _initialize_weights(self):
        # print(self.modules())

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)
                print(m.weight)

if __name__ == '__main__':
    # reading data
    dataReader = LoadData()

    max_epoch = 500     # max_epoch
    batch_size = 64         # batch size
    lr = 1e-4               # learning rate

    # define model
    model = Model()
    model._initialize_weights()     # init weight

    # loss and optimizer
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.01)    # param weight_decay is the L2 regularization

    num_train = dataReader.YTrain.shape[0]
    num_val = dataReader.YDev.shape[0]

    torch_dataset = TensorDataset(torch.FloatTensor(dataReader.XTrain), torch.LongTensor(dataReader.YTrain.reshape(num_train,)))
    XVal, YVal = torch.FloatTensor(dataReader.XDev), torch.LongTensor(dataReader.YDev.reshape(num_val,))
    train_loader = DataLoader(  # data loader class
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    et_acc = []        # store training loss
    ev_acc = []        # store validate loss

    for epoch in range(max_epoch):
        bt_acc = []  # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            pred = model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()         # backward
            optimizer.step()
            prediction = np.argmax(pred.cpu().data, axis=1)
            bt_acc.append(accuracy_score(batch_y.cpu().data, prediction))
        val_pred = np.argmax(model(XVal).cpu().data,axis=1)
        bv_acc = accuracy_score(dataReader.YDev,val_pred)
        et_acc.append(np.mean(bt_acc))
        ev_acc.append(bv_acc)
        print("Epoch: [%d / %d], Training Acc: %.6f, Val Acc: %.6f" % (epoch, max_epoch, np.mean(bt_acc), bv_acc))


    plt.plot([i for i in range(max_epoch)], et_acc)        # training loss
    plt.plot([i for i in range(max_epoch)], ev_acc)        # validate loss
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.show()







