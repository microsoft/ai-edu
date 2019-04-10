#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../CNN/MNIST_data/', one_hot=True)

learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step =  10

#宽度*迭代次数
n_inputs = 28
n_steps = 28
n_hidden = 256
n_classes = 10


x = tf.placeholder(tf.float32, [None, n_inputs*n_steps])
x_conv = tf.reshape(x, [-1, n_inputs, n_steps])
y = tf.placeholder(tf.float32, [None,n_classes])

#双向LSTM参数量加倍
weights = tf.Variable(tf.truncated_normal([2*n_hidden, n_classes], stddev=0.01))
bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))

def BiRNN(x, weights, bias):
    #(batch_size, n_steps, n_inputs)
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, n_inputs])
    x = tf.split(x, n_steps)
    
    #设置正向与反向lstm单元
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    #生成双向LSTM
   
    outputs, _ , _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    print('生成双向LSTM...')
    return tf.matmul(outputs[-1], weights) + bias

pred = BiRNN(x_conv, weights, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), dtype=tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    step = 1
    print('开始训练')
    while step*batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
       # batch_x = tf.reshape(batch_x, [batch_size, n_steps, n_inputs])
        optimizer.run({x:batch_x, y:batch_y})
        if step % display_step == 0:
            loss = cost.eval({x:batch_x, y:batch_y})
            acc = accuracy.eval({x:batch_x, y:batch_y})
            print('Iter'+ str(step*batch_size)+', Minibatch Loss=' + '{:.6f}'.format(loss) + ', Training Accuracy=' + '{:.5f}'.format(acc))
        step += 1
    print('Finish')
    
    test_len = 10000
    test_data = mnist.test.images[:test_len]
    test_label = mnist.test.labels[:test_len]
    
    print('Test Accuracy:', accuracy.eval({x:test_data, y:test_label}))




















