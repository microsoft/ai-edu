#encoding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np
import csv
import argparse

DNN_TRAINING = "G:\\TensorflowPJ\\train_data1.csv"
DNN_TEST = "G:\\TensorflowPJ\\test_data1.csv"
DNN_P = "G:\\TensorflowPJ\\test_data_4K_simple_37_unlabled.csv"
MODLE_FOLDER = "G:\TensorflowPJ\model_dir"



#读取测试数据
def get_test_inputs():
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DNN_TEST,
																	target_dtype=np.int,
																	features_dtype=np.float32)
	x = tf.constant(test_set.data)
	y = tf.constant(test_set.target)

	return x, y

def DNN():
	print("DNN Starting")
	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DNN_TRAINING,
																		target_dtype=np.int,
																		features_dtype=np.float32)

	# 设定37个特征
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=37)]

	# 建立40，80，40的三层深度神经网络
	classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,#把之前创建的特征列传入
												hidden_units=[40, 80, 40], #每层神经元数量，跟 DNN 原理有关。
												n_classes=2, #目标的类型的个数，目前是 2 个
												model_dir=MODLE_FOLDER) #训练模型保存的路径

	print("Training")
	#读取训练数据
	def get_train_inputs():
		x = tf.constant(training_set.data)
		y = tf.constant(training_set.target)
		return x, y
	#开始训练
	classifier.fit(input_fn=get_train_inputs, steps=1) #训练次数待调节

	print("Training completed")

	#测试准确度
	print("Test Strating")
	accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
	print("nTest Accuracy: {0:f}n".format(accuracy_score))
	
    #开始进行预测
	kinds = ["normal.", "attack"]
	predict_fp = open(DNN_P,'r',encoding = "UTF-8")
	matrix = [[0 for i in range(3)] for i in range(1000)]

	i = 0
	for line in predict_fp:
		line_list = line.strip().split(',')
		matrix[i] = line_list
		for j in range(37):
			matrix[i][j] = float(matrix[i][j])
		i = i + 1
	predict_set = np.array(matrix,dtype = float)
	predictions = list(classifier.predict(predict_set,as_iterable = True))
	for m in range(1000):
		print('Predictions: {}'.format(kinds[predictions[m]]))

if __name__ == '__main__':
	#就行DNN训练
	DNN()
