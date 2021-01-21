from HelperClass2.MnistImageDataReader import *
from HelperClass2.NeuralNet_3_0 import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import torch.nn.init as init
import warnings
warnings.filterwarnings('ignore')

train_data_name = "../../Data/ch11.train.npz"
test_data_name = "../../Data/ch11.test.npz"

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 64, bias=True)
        self.fc2 = nn.Linear(64, 16, bias=True)
        self.fc3 = nn.Linear(16, 10, bias=True)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)
                print(m.weight)

def metric(pred, label):
    '''

    :param pred: batch_size * num_classes, numpy array
    :param label: [batch_size,]
    :return: accuracy
    '''
    real_len = label.shape[0]
    pred_y = np.argmax(pred, axis=1)
    return sum(label == pred_y) / real_len

if __name__ == '__main__':
    # reading data
    dataReader = MnistImageDataReader(mode="vector")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=12)

    # data preprocessing
    dataReader.XTrain = np.reshape(dataReader.XTrain, [-1, 28 * 28])
    dataReader.YTrain = np.argmax(dataReader.YTrain, axis=1)
    dataReader.XDev = np.reshape(dataReader.XDev, [-1, 28 * 28])
    dataReader.YDev = np.argmax(dataReader.YDev, axis=1)

    max_epoch = 500         # max_epoch
    batch_size = 64         # batch size
    lr = 1e-4               # learning rate

    # define model
    model = Model()
    model._initialize_weights()     # init weight


    # loss and optimizer
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    num_train = dataReader.YTrain.shape[0]
    num_val = dataReader.YDev.shape[0]

    torch_dataset = TensorDataset(torch.FloatTensor(dataReader.XTrain), torch.LongTensor(dataReader.YTrain.reshape(num_train,)))
    XVal, YVal = torch.FloatTensor(dataReader.XDev), torch.LongTensor(dataReader.YDev.reshape(num_val,))
    train_loader = DataLoader(  # data loader class
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    et_loss = []        # store training loss
    ev_loss = []        # store validate loss

    for epoch in range(max_epoch):
        bt_loss = []  # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            pred = model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()         # backward
            optimizer.step()
            bt_loss.append(loss.cpu().data.numpy())
        val_pred = model(XVal)
        accuracy = metric(val_pred.cpu().data.numpy(), YVal.numpy())
        bv_loss = cross_entropy_loss(val_pred, YVal).cpu().data.numpy()
        et_loss.append(np.mean(bt_loss))
        ev_loss.append(bv_loss)
        print("Epoch: [%d / %d], Training Loss: %.6f, Val Loss: %.6f, Acc: %.6f" %
              (epoch, max_epoch, np.mean(bt_loss), bv_loss, accuracy))


    plt.plot([i for i in range(max_epoch)], et_loss)        # training loss
    plt.plot([i for i in range(max_epoch)], ev_loss)        # validate loss
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.show()







