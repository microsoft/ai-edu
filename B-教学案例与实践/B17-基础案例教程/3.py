#_*_coding:utf8 _*_ 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy import float32
#将该session注册为默认的session, 之后的运算默认在此session中进行

mnist = input_data.read_data_sets("../CNN/MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#创建一个placeholder 即输入数据的地方
#placeholder第一个参数是数据类型，第二个参数代表tensor的shape,即数据的尺寸.
x = tf.placeholder(tf.float32, [None,784]);
#使用Variable对象来存储模型参数
#将weights和biases全部初始化为0,模型训练时会自动学习合适的值
#对于卷积网络，循环网络来说初始化的值至关重要。
W = tf.Variable(tf.zeros([784,10]));
b = tf.Variable(tf.zeros([10]))
#实现softmax算法,forward和backward内容自动实现。
#matmul矩阵乘法
y = tf.nn.softmax(tf.matmul(x,W)+b)
#对于分类问题可以使用对数似然函数求得熵值
#真实的分类值来计算代价损失
y_ = tf.placeholder(tf.float32, [None,10])
print(y_.get_shape())
#y_*tf.log(y) 代表公式中的 y'log(y)  tf.reduce_mean代表求和  tf.reduce_mean表示对每个batch结果求均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y), reduction_indices=[1]))
#定义优化算法 随机梯度下降SGD
#设置学习速率为0.5 优化目标设定为cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy);

#调用TensorFlow初始化器
tf.global_variables_initializer().run()

#迭代执行训练操作，每次随机从训练集中取100条样本构成一个minibatch并feed给placeholder
#使用一小部分的样本进行训练成为随机梯度下降
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100);
    train_step.run(feed_dict = {x:batch_xs,y_:batch_ys})

#对模型的准确率进行验证
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy)
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))



writer  = tf.summary.FileWriter(r'D:/tf', tf.get_default_graph())
writer.close()











