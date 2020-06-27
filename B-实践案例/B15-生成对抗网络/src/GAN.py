# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import os

from PIL import Image

from HelperClass2.MnistImageDataReader_2_0 import *
from HelperClass2.LossFunction_1_2 import *
from HelperClass2.ActivatorFunction_2_0 import *
from HelperClass2.ClassifierFunction_2_0 import *
from HelperClass2.WeightsBias_1_0 import *

class GAN(object):
  def __init__(self, model_name):
    self.model_name = model_name
    self.subfolder = os.getcwd() + "/" + self.__create_subfolder()

    self.init_method = InitialMethod.MSRA
    self.eta = 0.01

    # 判别器三层网络参数
    self.d_wb1 = WeightsBias_1_0(784, 512, self.init_method, self.eta)
    self.d_wb1.InitializeWeights(self.subfolder, True)
    self.d_wb2 = WeightsBias_1_0(512, 256, self.init_method, self.eta)
    self.d_wb2.InitializeWeights(self.subfolder, True)
    self.d_wb3 = WeightsBias_1_0(256, 1, self.init_method, self.eta)
    self.d_wb3.InitializeWeights(self.subfolder, True)

    # 生成器三层网络参数
    self.g_wb1 = WeightsBias_1_0(100, 256, self.init_method, self.eta)
    self.g_wb1.InitializeWeights(self.subfolder, True)
    self.g_wb2 = WeightsBias_1_0(256, 512, self.init_method, self.eta)
    self.g_wb2.InitializeWeights(self.subfolder, True)
    self.g_wb3 = WeightsBias_1_0(512, 784, self.init_method, self.eta)
    self.g_wb3.InitializeWeights(self.subfolder, True)

  def __create_subfolder(self):
    if self.model_name != None:
      path = self.model_name.strip()
      path = path.rstrip("/")
      isExists = os.path.exists(path)
      if not isExists:
        os.makedirs(path)
      return path

  def d_forward(self, batch_x):
    # 判别器前向计算，前两层激活函数为Relu，最后一层为二分类
    self.d_Z1 = np.dot(batch_x, self.d_wb1.W) + self.d_wb1.B
    self.d_A1 = Relu().forward(self.d_Z1)
    self.d_Z2 = np.dot(self.d_A1, self.d_wb2.W) + self.d_wb2.B
    self.d_A2 = Relu().forward(self.d_Z2)
    self.d_Z3 = np.dot(self.d_A2, self.d_wb3.W) + self.d_wb3.B
    self.d_A3 = Logistic().forward(self.d_Z3)
    return self.d_A3
    
  def g_forward(self, batch_x):
    # 生成器前向计算，前两层激活函数为Relu，最后一层为Tanh
    # 输入长度100的随机数，输出长度784，即生成一张28*28的图片
    self.g_Z1 = np.dot(batch_x, self.g_wb1.W) + self.g_wb1.B
    self.g_A1 = Relu().forward(self.g_Z1)
    self.g_Z2 = np.dot(self.g_A1, self.g_wb2.W) + self.g_wb2.B
    self.g_A2 = Relu().forward(self.g_Z2)
    self.g_Z3 = np.dot(self.g_A2, self.g_wb3.W) + self.g_wb3.B
    self.g_A3 = Tanh().forward(self.g_Z3)
    return self.g_A3

  def d_backward(self, batch_x, batch_y, batch_output):
    m = batch_x.shape[0]

    # 对判别器的各层进行反向传播，并计算各层的梯度
    dZ3 = batch_output - batch_y
    self.d_wb3.dW = np.dot(self.d_A2.T, dZ3)/m
    self.d_wb3.dB = np.sum(dZ3, axis=0, keepdims=True)/m

    dA2 = np.dot(dZ3, self.d_wb3.W.T)
    dZ2,_ = Relu().backward(self.d_Z2, self.d_A2, dA2)
    self.d_wb2.dW = np.dot(self.d_A1.T, dZ2)/m
    self.d_wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m

    dA1 = np.dot(dZ2, self.d_wb2.W.T) 
    dZ1,_ = Relu().backward(self.d_Z1, self.d_A1, dA1)
    self.d_wb1.dW = np.dot(batch_x.T, dZ1)/m
    self.d_wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m

  def d_update(self):
    # 根据反向传播中计算的梯度来更新判别器参数
    self.d_wb1.Update()
    self.d_wb2.Update()
    self.d_wb3.Update()

  def g_backward(self, batch_x, batch_y, batch_output):
    m = batch_x.shape[0]

    # 对判别器的各层进行反向传播，但此处不需要更新判别器，所以不计算梯度
    dZ3 = batch_output - batch_y
    dA2 = np.dot(dZ3, self.d_wb3.W.T)
    dZ2,_ = Relu().backward(self.d_Z2, self.d_A2, dA2)
    dA1 = np.dot(dZ2, self.d_wb2.W.T)
    dZ1,_ = Relu().backward(self.d_Z1, self.d_A1, dA1)
    
    # 对生成器的各层进行反向传播，并计算各层的梯度
    dA3 = np.dot(dZ1, self.d_wb1.W.T)
    dZ3,_ = Tanh().backward(None, self.g_A3, dA3)
    self.g_wb3.dW = np.dot(self.g_A2.T, dZ3)/m
    self.g_wb3.dB = np.sum(dZ3, axis=0, keepdims=True)/m

    dA2 = np.dot(dZ3, self.g_wb3.W.T)
    dZ2,_ = Relu().backward(self.g_Z2, self.g_A2, dA2)
    self.g_wb2.dW = np.dot(self.g_A1.T, dZ2)/m
    self.g_wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m

    dA1 = np.dot(dZ2, self.g_wb2.W.T)
    dZ1,_ = Relu().backward(self.g_Z1, self.g_A1, dA1)
    self.g_wb1.dW = np.dot(batch_x.T, dZ1)/m
    self.g_wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m

  def g_update(self):
    # 根据反向传播中计算的梯度来更新生成器参数
    self.g_wb1.Update()
    self.g_wb2.Update()
    self.g_wb3.Update()


def save_imgs(gan, name):
  output_folder = 'GAN_output'
  os.makedirs(output_folder, exist_ok=True)

  random_input = np.random.normal(size = (16, 100))
  fakes = gan.g_forward(random_input)

  # [-1, 1] => [0, 1] => [0, 255]
  imgs = np.uint8(np.floor(((fakes + 1)/ 2) * 255 + 0.5))

  # 784 => 28 * 28
  imgs=list(map(lambda x: np.reshape(x, (28,28)), imgs))

  # 4 row * 4 column
  bigimage = Image.fromarray(np.vstack(list(map(lambda r: np.hstack(imgs[r*4:r*4+4]), range(4)))))

  bigimage = bigimage.convert('L')
  bigimage.save(os.path.join(output_folder, str(name) + '.png'))


if __name__ == '__main__':
  dataReader = MnistImageDataReader_2_0(mode="vector")
  dataReader.ReadData()
  dataReader.NormalizeX()
  dataReader.Shuffle()

  gan = GAN('GAN_MNIST')

  max_epoch = 200
  batch_size = 64
  max_iteration = np.ceil(dataReader.num_train / batch_size).astype(int)

  loss_func = LossFunction_1_2(NetType.BinaryClassifier)

  total_iteration = 0
  for epoch in range(max_epoch):
    dataReader.Shuffle()
    for iteration in range(max_iteration):
      # 保存16个生成的假样本到图片
      if total_iteration % 1000 == 0:
        save_imgs(gan, total_iteration)

      # 真样本
      real_x, _ = dataReader.GetBatchTrainSamples(batch_size, iteration)
      current_batch_size = real_x.shape[0]
      
      # 随机产生生成器的输入
      g_random_input = np.random.normal(size = (current_batch_size,100))

      # 生成器产生假样本
      fake_x = gan.g_forward(g_random_input)

      # 将真假样本一起输入到判别器
      d_input = np.append(real_x, fake_x, axis=0)
      d_output = gan.d_forward(d_input)
      d_label = np.append(np.ones((current_batch_size,1)), np.zeros((current_batch_size,1)), axis=0)

      # 判别器反向传播并更新参数
      gan.d_backward(d_input, d_label, d_output)
      gan.d_update()

      # 计算loss用于结果分析
      d_loss = loss_func.CheckLoss(d_output, d_label)
      
      # 用更新过参数的判别器重新判别假样本
      d_out_fake = gan.d_forward(fake_x)

      # 生成器反向传播并更新参数
      gan.g_backward(g_random_input, np.ones((current_batch_size, 1)), d_out_fake)
      gan.g_update()
      
      # 计算loss用于结果分析
      g_loss = loss_func.CheckLoss(d_out_fake, np.ones((current_batch_size, 1)))
      print(epoch, iteration, total_iteration, d_loss, g_loss)

      total_iteration += 1

  save_imgs(gan, 'final')

  print('done')