import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from torch.utils.data import TensorDataset, DataLoader
from HelperClass.NeuralNet_1_2 import *
from HelperClass.Visualizer_1_0 import *
from Level2_ShowMultipleResult import ShowData
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

file_name = "../../data/ch07.npz"

def ShowResult(W, B, X, Y, xt, yt):
    fig = plt.figure(figsize=(6,6))
    DrawThreeCategoryPoints(X[:,0], X[:,1], Y[:], xlabel="x1", ylabel="x2", show=False)

    b13 = (B[0,0] - B[0,2])/(W[1,2] - W[1,0])
    w13 = (W[0,0] - W[0,2])/(W[1,2] - W[1,0])

    b23 = (B[0,2] - B[0,1])/(W[1,1] - W[1,2])
    w23 = (W[0,2] - W[0,1])/(W[1,1] - W[1,2])

    b12 = (B[0,1] - B[0,0])/(W[1,0] - W[1,1])
    w12 = (W[0,1] - W[0,0])/(W[1,0] - W[1,1])

    x = np.linspace(0,1,2)
    y = w13 * x + b13
    p13, = plt.plot(x,y,c='r')

    x = np.linspace(0,1,2)
    y = w23 * x + b23
    p23, = plt.plot(x,y,c='b')

    x = np.linspace(0,1,2)
    y = w12 * x + b12
    p12, = plt.plot(x,y,c='g')

    plt.legend([p13,p23,p12], ["13","23","12"])
    plt.axis([-0.1,1.1,-0.1,1.1])

    DrawThreeCategoryPoints(xt[:,0], xt[:,1], yt[:], xlabel="x1", ylabel="x2", show=True, isPredicate=True)

class Model(nn.Module):
    def __init__(self, input_size, class_num):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, class_num)
    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x)
        return x

if __name__ == '__main__':
    max_epoch = 500
    num_category = 3
    reader = DataReader_1_3(file_name)
    reader.ReadData()
    # show raw data before normalization
    reader.NormalizeX()

    num_input = 2       # input size
    # get numpy form data
    XTrain, YTrain = reader.XTrain, reader.YTrain - 1
    torch_dataset = TensorDataset(torch.FloatTensor(XTrain), torch.LongTensor(YTrain.reshape(-1,)))
    reader.ToOneHot(num_category, base=1)       # transform to one-hot
    ShowData(reader.XRaw, reader.YTrain)
    train_loader = DataLoader(          # data loader class
        dataset=torch_dataset,
        batch_size=32,
        shuffle=False,
    )
    loss_func = nn.CrossEntropyLoss()
    model = Model(num_input,num_category)
    optimizer = Adam(model.parameters(), lr=1e-2)


    e_loss = []     # mean loss at every epoch
    for epoch in range(max_epoch):
        b_loss = []     # mean loss at every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_func(pred,batch_y)
            loss.backward()
            optimizer.step()
            b_loss.append(loss.cpu().data.numpy())
        print("Epoch: %d, Loss: %.5f" % (epoch, np.mean(b_loss)))

    xt_raw = np.array([5, 1, 7, 6, 5, 6, 2, 7]).reshape(4, 2)
    xt = reader.NormalizePredicateData(xt_raw)
    xt = torch.FloatTensor(xt)
    output = model(xt).cpu().data.numpy()

    ShowResult(model.fc.weight.data.numpy().transpose(), model.fc.bias.data.numpy().reshape(1, 3),
               reader.XTrain, reader.YTrain, xt, output)

