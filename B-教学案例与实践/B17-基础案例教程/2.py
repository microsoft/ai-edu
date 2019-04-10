# _*_ coding:utf8 _*_
'''
Created on 2017年12月6日

@author: knight
'''

from datetime import datetime
import math
import time 
import tensorflow as tf


batch_size = 10
num_batches = 100

#展示每一个卷积层或者池化层输出tensor的尺寸

def print_activation(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def full_connect(input, num_in, num_out, drop_out=1.0):
    w = tf.Variable(tf.truncated_normal([num_in, num_out]))
    b = tf.Variable(tf.truncated_normal([num_out]))
    return tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w), b)), keep_prob = drop_out)
    
#定义AlexNet网络结构
#定义inference 接收 images作为输入 返回最后一层pool5和parameters(网络中所有需要训练的模型参数)

def inference(images):
    parameters = []
    
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64], stddev=1e-1, dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(images, kernel, strides=[1,4,4,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activation(conv1)
        parameters += [kernel, biases]
        
    #感觉lrn层意义不太大
    lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')
#     pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,4,4,1], padding='VALID', name='pool1')
    print_activation(pool1)
        
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192], stddev=1e-1, dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    
    print_activation(conv2)
    lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
#     pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
    print_activation(pool2)
    
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384], stddev=1e-1, dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        
    print_activation(conv3)
    
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=1e-1, dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype = tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activation(conv4)
    
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=1e-1, dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    
    print_activation(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')
    print_activation(pool5)
    
    flatten = tf.reshape(pool5, [-1, 6*6*256])
    fc_1 = full_connect(flatten, 6*6*256, 4096, 0.5)
    fc_2 = full_connect(fc_1, 4096,4096, 0.5)
    fc_3 = full_connect(fc_2, 4096,1000)
    return fc_3, parameters
        #卷积层已经搭建完毕,如果想正式使用AlexNet还需另外添加三个全连接层(4096,4096,1000)使用dropout


#评估AlexNet每轮的计算时间
#第一个参数是Session, 第二个是需要运算的tensor, 第三个是名称
def time_tensorflow_run(session, target, info_string):
    #预热轮数 增加cache 
    num_steps_burn_in = 10
    #总时间
    total_duration = 0.0
    #计算方差
    total_duration_squared = 0.0
    
    for i in range(num_steps_burn_in + num_batches):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i>= num_steps_burn_in:
            if not i % 10:
                print(r'%s: step:%d. duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    
    #每轮迭代的平均耗时
    mn = total_duration / num_batches
    #标准差
    vr = total_duration_squared / num_batches - mn*mn
    sd = math.sqrt(vr)
    print(r'%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))
    
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3], dtype=tf.float32, stddev=1e-1))
    
        pool5, parameters = inference(images)
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        #进行前馈计算
        time_tensorflow_run(sess, pool5, "forward")
        
        objective = tf.nn.l2_loss(pool5)
        #tf.gradients求相对于loss的梯度参数
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'Forward-backword')
        

run_benchmark()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
