# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import numpy as np
import os
import shutil
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from conv2d import Cconv2d
from pool import Cpool
from relu import Crelu
from softmax import Csoftmax
from fc import Cfc
from dropout import Cdropout
from save import model_save
from functools import reduce


FLAGS = None

# SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = None
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 2
VALIDATION_SIZE = 500  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
TRAINING_SIZE = 12665
TESTING_SIZE = 2115
SOURCE_FOLDER = 'D:\\practice\\AI\\samples-for-ai\\examples\\tensorflow\\MNIST\\Data\\'


class Creshape(object):
  def __init__(self, shape, name="", exname=""):
    self.type = "Reshape"
    self.shape = shape
    self.input_name = exname
    self.output_name = name

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32

def copy_file(filename):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        tf.gfile.Copy(SOURCE_FOLDER + filename, filepath)
    return filepath

def extract_image_data(filename, num_images):
    with open(filename, 'rb') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH/2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        np.save("img.npy", data[1:10])
        return data

def extract_labels_data(filename, num_images):
    print('Extracting', filename)
    with open(filename, 'rb') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        return labels

def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(_):
  global BATCH_SIZE
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = copy_file('train-images-01')
    train_labels_filename = copy_file('train-labels-01')
    test_data_filename = copy_file('test-images-01')
    test_labels_filename = copy_file('test-labels-01')

    # Extract it into numpy arrays.
    train_data = extract_image_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels_data(train_labels_filename, TRAINING_SIZE)
    test_data = extract_image_data(test_data_filename, TESTING_SIZE)
    test_labels = extract_labels_data(test_labels_filename, TESTING_SIZE)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

  train_size = train_labels.shape[0]

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  class Cmodel(object):
    def __init__(self, train_size):
        """The Model definition."""
        self.conv1 = Cconv2d([train_size, 28, 28, 1], 5, 32, name="conv1", exname="")
        self.relu1 = Crelu(self.conv1.outputShape, name="relu1", exname="conv1")
        self.pool1 = Cpool(self.relu1.shape, name="pool1", exname="relu1")
        self.conv2 = Cconv2d(self.pool1.outputShape, 5, 64, name="conv2", exname="pool1")
        self.relu2 = Crelu(self.conv2.outputShape, name="relu2", exname="conv2")
        self.pool2 = Cpool(self.relu2.shape, name="pool2", exname="relu2")
        self.fc1 = Cfc(self.pool2.outputShape, 512, name="fc1", exname="reshape")
        self.relu3 = Crelu(self.fc1.outputShape, name="relu3", exname="fc1")
        self.dropout = Cdropout(self.relu3.shape, prob=0.5)
        self.fc2 = Cfc(self.dropout.outputShape, NUM_LABELS, name="fc2", exname="relu3")
        self.softmax = Csoftmax([train_size, NUM_LABELS], name="softmax", exname="fc2")

        self.model = [
          self.conv1, self.relu1, self.pool1, self.conv2, self.relu2, self.pool2, 
          Creshape([1, reduce(lambda x, y: x * y, self.fc1.input_size[1:])], name=self.fc1.input_name, exname=self.pool2.output_name),
          self.fc1, self.relu3, self.fc2, self.softmax
        ]
    
    def forward(self, data, train=False):
        net = self.conv1.forward(data)
        net = self.relu1.forward(net)
        net = self.pool1.forward(net)
        net = self.conv2.forward(net)
        net = self.relu2.forward(net)
        net = self.pool2.forward(net)
        net = self.fc1.forward(net)
        net = self.relu3.forward(net)
        net = self.dropout.forward(net, train=train)
        net = self.fc2.forward(net)
        return net
    
    def calLoss(self, labels, perdictions):
        loss = self.softmax.calLoss(labels, perdictions)
        return loss
    
    def gradient(self):
        error = self.softmax.gradient()
        error = self.fc2.gradient(error)
        error = self.dropout.gradient(error)
        error = self.relu3.gradient(error)
        error = self.fc1.gradient(error)
        error = self.pool2.gradient(error)
        error = self.relu2.gradient(error)
        error = self.conv2.gradient(error)
        error = self.pool1.gradient(error)
        error = self.relu1.gradient(error)
        error = self.conv1.gradient(error)

    def backward(self, learningRate=0.00001):
        self.gradient()
        self.conv1.backward(learningRate=learningRate)
        self.conv2.backward(learningRate=learningRate)
        self.fc1.backward(learningRate=learningRate)
        self.fc2.backward(learningRate=learningRate)

  def eval_in_batches(model, data):
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = model.forward(data[begin:end, ...])
      else:
        batch_predictions = model.forward(data[-EVAL_BATCH_SIZE:, ...])
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
  
  best_loss = 1e6
  mnistModel = Cmodel(BATCH_SIZE)
  for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    learningRate = 0.0001 * (0.95 ** ((step * BATCH_SIZE) // train_size))
    # learningRate = 0.0001
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    predictions = mnistModel.forward(batch_data, train=True)
    loss = mnistModel.calLoss(batch_labels, predictions)
    print(loss)
    mnistModel.backward(learningRate)
    if step % EVAL_FREQUENCY == 0:
      print('Step %d (epoch %.2f)' %
              (step, float(step) * BATCH_SIZE / train_size))
      print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
      print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(mnistModel, validation_data), validation_labels))
      print('learning rate: %.6f' % learningRate)
      if(loss <= best_loss):
        best_loss = loss
        np.save("conv1_w.npy", mnistModel.conv1.weights)
        model_save(mnistModel.model, "./output/")
        
    
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')
  parser.add_argument(
    '--input_dir',
    type=str,
    default='input',
    help='Directory to put the input data.')
  parser.add_argument(
    '--model_dir',
    type=str,
    default='output',
    help='Directory to put the checkpoint files.')
  parser.add_argument(
    '--export_dir',
    type=str,
    default='export',
    help='Directory to put the savedmodel files.')

  FLAGS, unparsed = parser.parse_known_args()
  WORK_DIRECTORY = FLAGS.input_dir
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
