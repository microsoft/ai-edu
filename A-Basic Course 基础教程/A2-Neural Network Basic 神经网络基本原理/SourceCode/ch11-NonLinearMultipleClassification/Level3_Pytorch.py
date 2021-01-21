from HelperClass2.NeuralNet_2_2 import *
from HelperClass2.Visualizer_1_1 import *
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
        self.fc1 = nn.Linear(2, 3, bias=True)
        self.fc2 = nn.Linear(3, 3, bias=True)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x))
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
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.YTrain = dataReader.YTrain - 1       # (1, 2, 3) ----> (0, 1, 2)
    dataReader.YDev = dataReader.YDev - 1

    # print(dataReader.YTrain.shape)

    max_epoch = 500     # max_epoch
    batch_size = 64         # batch size
    lr = 0.05               # learning rate

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
        bv_loss = cross_entropy_loss(val_pred, YVal).cpu().data.numpy()
        et_loss.append(np.mean(bt_loss))
        ev_loss.append(bv_loss)
        print("Epoch: [%d / %d], Training Loss: %.6f, Val Loss: %.6f" % (epoch, max_epoch, np.mean(bt_loss), bv_loss))


    plt.plot([i for i in range(max_epoch)], et_loss)        # training loss
    plt.plot([i for i in range(max_epoch)], ev_loss)        # validate loss
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.show()







