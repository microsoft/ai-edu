# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet_4_2 import *
from ExtendedDataReader.MnistImageDataReader import *

train_x = '../../Data/FashionMnistTrainX'
train_y = '../../Data/FashionMnistTrainY'
test_x = '../../Data/FashionMnistTestX'
test_y = '../../Data/FashionMnistTestY'

# 0-T恤 1-裤子 2-套衫 3-连衣裙 4-外套 5-凉鞋 6-衬衫 7-运动鞋 8-包 9-短靴
names=["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def LoadData(mode):
    mdr = MnistImageDataReader(train_x, train_y, test_x, test_y, mode)
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr

def dnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    bn1 = BnLayer(f1.output_size)
    net.add_layer(bn1, "bn1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    bn2 = BnLayer(f2.output_size)
    net.add_layer(bn2, "bn2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net

def show_result(x,y,z,title,mode):
    fig,ax = plt.subplots(nrows=6, ncols=6, figsize=(9,10))
    for i in range(36):
        if mode == "vector":
            ax[i//6,i%6].imshow(x[i].reshape(28,28), cmap='gray')
        else:
            ax[i//6,i%6].imshow(x[i,0], cmap='gray')
        if y[i] == z[i]:
            ax[i//6,i%6].set_title(names[z[i]])
        else:
            ax[i//6,i%6].set_title("*" + names[z[i]] + "(" + str(y[i]) + ")")
        ax[i//6,i%6].axis('off')
    #endfor
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    mode = "vector"
    dataReader = LoadData(mode)
    net = dnn_model()
    net.train(dataReader, checkpoint=0.1, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)

    X_test,Y_test = dataReader.GetTestSet()
    count = 36
    X = X_test[0:count]
    Y = Y_test[0:count]
    Z = net.inference(X)
    show_result(X,np.argmax(Y,axis=1),np.argmax(Z,axis=1),"dnn predication",mode)
