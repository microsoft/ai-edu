# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from HelperClass2.MnistImageDataReader import *
from HelperClass2.NeuralNet_3_0 import *

def model(eta, batch_size):

    filename = str.format("LR_BS_Trial\\loss_{0}_{1}.pkl", eta, batch_size).replace('.','',1)
    filepath = Path(filename)
    if filepath.exists():
        return filename

    dataReader = MnistImageDataReader(mode="vector")
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=12)

    n_input = dataReader.num_feature
    n_hidden1 = 64
    n_hidden2 = 16
    n_output = dataReader.num_category
    eps = 0.01
    max_epoch = 30

    hp = HyperParameters_3_0(
        n_input, n_hidden1, n_hidden2, n_output, 
        eta, max_epoch, batch_size, eps, 
        NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet_3_0(hp, "LR_BS_Trial")
    net.train(dataReader, 0.5, True)
    net.DumpLossHistory(filename)
    return filename

if __name__ == '__main__':
    file1 = model(0.2, 128)
    file2 = model(0.3, 128)
    file3 = model(0.5, 128)
    file4 = model(0.8, 128)

    th = TrainingHistory_2_3.Load(file1)
    p1, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file2)
    p2, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file3)
    p3, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file4)
    p4, = plt.plot(th.iteration_seq, th.accuracy_val)
    plt.legend([p1,p2,p3,p4], ["0.2","0.3","0.5","0.8"])
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title("accuracy on learning rate, batch_size=128")
    plt.show()

    file1 = model(0.5, 32)
    file2 = model(0.5, 64)
    file3 = model(0.5, 128)
    file4 = model(0.5, 256)

    th = TrainingHistory_2_3.Load(file1)
    p1, = plt.plot(th.epoch_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file2)
    p2, = plt.plot(th.epoch_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file3)
    p3, = plt.plot(th.epoch_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file4)
    p4, = plt.plot(th.epoch_seq, th.accuracy_val)
    plt.legend([p1,p2,p3,p4], ["32","64","128","256"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy on batch size, eta=0.5")
    plt.show()

    file1 = model(0.1, 32)
    file2 = model(0.3, 32)
    file3 = model(0.5, 32)
    file4 = model(0.7, 32)

    th = TrainingHistory_2_3.Load(file1)
    p1, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file2)
    p2, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file3)
    p3, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file4)
    p4, = plt.plot(th.iteration_seq, th.accuracy_val)
    plt.legend([p1,p2,p3,p4], ["0.1","0.3","0.5","0.7"])
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title("accuracy on learning rate, batch size=32")
    plt.show()

    file1 = model(0.1, 16)
    file2 = model(0.3, 16)
    file3 = model(0.5, 16)
    file4 = model(0.7, 16)

    th = TrainingHistory_2_3.Load(file1)
    p1, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file2)
    p2, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file3)
    p3, = plt.plot(th.iteration_seq, th.accuracy_val)
    th = TrainingHistory_2_3.Load(file4)
    p4, = plt.plot(th.iteration_seq, th.accuracy_val)
    plt.legend([p1,p2,p3,p4], ["0.1","0.3","0.5","0.7"])
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title("accuracy on learning rate, batch size=16")
    plt.show()
