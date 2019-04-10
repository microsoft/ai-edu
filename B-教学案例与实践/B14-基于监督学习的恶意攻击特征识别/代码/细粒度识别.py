'''
基于tensorflow可识别多种网络攻击的有监督学习程序
'''
from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe


tf.enable_eager_execution()  #启动eager execution动态图模块

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_fp = "G:\\TensorflowPJ\\Keras_N\\train_data_4K_simple_38.csv"



def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.],[0.], [0.], [0.],
  [0.],[0.], [0.], [0.],[0.], [0.], [0.],[0.], [0.], [0.],[0.], [0.], [0.],[0.], [0.],
   [0.],[0.], [0.], [0.],[0.], [0.], [0.],[0.], [0.], [0.],[0.], [0.],[0.], [0]]  # 设置数据格式
  parsed_line = tf.decode_csv(line, example_defaults)
  # 前37列是特征值 构成一个TENSOR 来自于KDD CUP 99数据集
  features = tf.reshape(parsed_line[:-1], shape=(38,))
  #最后一列是数据标签 代表不同的攻击（目前有4种）
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

#创建训练用的dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)             
train_dataset = train_dataset.map(parse_csv)      
#train_dataset = train_dataset.shuffle(buffer_size=3001)  #设置一个缓冲区 将样本进行随机化处理 缓冲区大小要大于数据集总数
train_dataset = train_dataset.batch(32)  #批处理设置 加快运行速度


#设置神经网络模型 此处使用了Keras库中的顺序模型Sequential 激活函数为relu 两个隐藏层分别有50个神经元节点
model = tf.keras.Sequential([
  tf.keras.layers.Dense(50, activation="relu", input_shape=(38,)),  
  tf.keras.layers.Dense(50, activation="relu"),
  tf.keras.layers.Dense(5)
])

#定义损失函数
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

#定义梯度下降函数
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) #学习率 即为梯度下降的步长

train_loss_results = []
train_accuracy_results = []

num_epochs = 6   #迭代次数
#进行迭代训练
for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  for x, y in train_dataset:
    grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    epoch_loss_avg(loss(model, x, y))
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 1 == 0:
    
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

#做出损失率和成功率随迭代次数变化的图像
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()

#设置测试集
test_fp = "G:\\TensorflowPJ\\Keras_N\\test_data_4K.csv"


test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             
test_dataset = test_dataset.map(parse_csv)      
#test_dataset = test_dataset.shuffle(1001)      
test_dataset = test_dataset.batch(10000)           

test_accuracy = tfe.metrics.Accuracy()
#进行测试
for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

#对新数据集进行预测
class_ids = ["normal.", "ipsweep.", "neptune.","smurf.","portsweep."] #标明不同的id对应哪种攻击方式的列表
predict_fp = open("G:\\TensorflowPJ\\Keras_N\\test_data_4K_simple_38_unlabled.csv",'r',encoding = "UTF-8")
matrix = [[0 for i in range(3)] for i in range(1000)]
matrix_c = [[0 for i in range(3)] for i in range(1000)]

i = 0
for line in predict_fp:
  line_list = line.strip().split(',')
  matrix[i] = line_list
  for j in range(38):
    matrix[i][j] = float(matrix[i][j])
  i = i + 1


predict_dataset = tf.convert_to_tensor(matrix)
predictions = model(predict_dataset) 

compare_fp = open("G:\\TensorflowPJ\\Keras_N\\test_data_4K_simple_38_labled.csv",'r',encoding = "UTF-8")

m = 0
for line1 in compare_fp:
  line_list_1 = line1.strip().split(',')
  matrix_c[m] = line_list_1
  for n in range(39):
    matrix_c[m][n] = float(matrix_c[m][n])
  m = m + 1

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  name = class_ids[class_idx]
  if class_idx == matrix_c[i][38]:
    print("Example {} prediction: {}".format(i, name),"  true")
  else:
    print("Example {} prediction: {}".format(i, name),"  false")

