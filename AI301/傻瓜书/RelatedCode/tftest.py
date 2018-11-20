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



FLAGS = None

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = None
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 30
NUM_EPOCHS = 100
EVAL_BATCH_SIZE = 30
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
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

# @profile
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
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    # train_data = train_data[VALIDATION_SIZE:, ...]
    # train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      name='image_input')
  train_labels_node = tf.placeholder(tf.int64, shape=(None,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
#   conv1_weights = tf.Variable(
#       tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
#                           stddev=0.1,
#                           seed=SEED, dtype=data_type()))
#   conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
#   conv2_weights = tf.Variable(tf.truncated_normal(
#       [5, 5, 32, 64], stddev=0.1,
#       seed=SEED, dtype=data_type()))
#   conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, 392],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[392], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([392, 64],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[64], dtype=data_type()))
  fc3_weights = tf.Variable(tf.truncated_normal([64, 32],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc3_biases = tf.Variable(tf.constant(
      0.1, shape=[32], dtype=data_type()))
  fc4_weights = tf.Variable(tf.truncated_normal([32, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc4_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    # conv = tf.nn.conv2d(data,
    #                     conv1_weights,
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME')
    # # Bias and rectified linear non-linearity.
    # relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # # Max pooling. The kernel size spec {ksize} also follows the layout of
    # # the data. Here we have a pooling window of 2, and a stride of 2.
    # pool = tf.nn.max_pool(relu,
    #                       ksize=[1, 2, 2, 1],
    #                       strides=[1, 2, 2, 1],
    #                       padding='SAME')
    # conv = tf.nn.conv2d(pool,
    #                     conv2_weights,
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME')
    # relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    # pool = tf.nn.max_pool(relu,
    #                       ksize=[1, 2, 2, 1],
    #                       strides=[1, 2, 2, 1],
    #                       padding='SAME')
    # # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # # fully connected layers.
    # pool_shape = pool.get_shape().as_list()
    # reshape = tf.reshape(
    #     pool,
    #     [tf.shape(pool)[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    data_shape = data.get_shape().as_list()
    data_reshape = tf.reshape(data, [BATCH_SIZE, data_shape[1] * data_shape[2] * data_shape[3]])
    hidden = tf.nn.relu(tf.matmul(data_reshape, fc1_weights) + fc1_biases)
    hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)
    hidden = tf.nn.relu(tf.matmul(hidden, fc3_weights) + fc3_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    # if train:
    #   hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc4_weights) + fc4_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))
  predict_op = tf.argmax(logits, 1, name='predict_op')

  # L2 regularization for the fully connected parameters.
  # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
  #                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  # loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  # learning_rate = tf.train.exponential_decay(
  #     0.01,                # Base learning rate.
  #     batch * BATCH_SIZE,  # Current index into the dataset.
  #     train_size,          # Decay step.
  #     0.95,                # Decay rate.
  #     staircase=True)
  learning_rate = tf.constant(0.005)
  # Use simple momentum for the optimization.
  # optimizer = tf.train.MomentumOptimizer(0.005,
  #                                        0.9).minimize(loss,
  #                                                      global_step=batch)
  
  optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  ### Change original code
  # Add model_dir to save model
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

  ### Change original code
  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()
  # Create a builder for writing saved model for serving.
  if os.path.isdir(FLAGS.export_dir):
    shutil.rmtree(FLAGS.export_dir)
  builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.export_dir)


  # Create a local session to run the training.
  
  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    ### Change original code
    # Save checkpoint when training
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print('Load from ' + ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)

    ### Change original code
    # Create summary, logs will be saved, which can display in Tensorboard
    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, 'log'), sess.graph)

    print('Initialized!')
    start = time.time()
    # Loop through training steps.
    start_time = time.time()
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)
      # print some extra information once reach the evaluation frequency
      # if step % EVAL_FREQUENCY == 0:
      #   # fetch some extra nodes' data
      #   ### Change original code
      #   # Add summary
      #   summary, l, lr, predictions = sess.run([merged, loss, learning_rate, train_prediction],
      #                                 feed_dict=feed_dict)
      #   writer.add_summary(summary, step)
      #   elapsed_time = time.time() - start_time
      #   start_time = time.time()
      #   ### Change original code
      #   # save model
      #   if step % (EVAL_FREQUENCY * 10) == 0:
      #     saver.save(sess, os.path.join(FLAGS.model_dir, "model.ckpt"), global_step=step)
      #   print('Step %d (epoch %.2f), %.1f ms' %
      #         (step, float(step) * BATCH_SIZE / train_size,
      #          1000 * elapsed_time / EVAL_FREQUENCY))
      #   print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
      #   print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
      #   print('Validation error: %.1f%%' % error_rate(
      #       eval_in_batches(validation_data, sess), validation_labels))
      #   sys.stdout.flush()

    end = time.time()
    print(end - start)


    ### Change original code
    # Save model
    inputs = { tf.saved_model.signature_constants.PREDICT_INPUTS: train_data_node }
    outputs = { tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_op }
    serving_signatures = {
      'Infer': #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      tf.saved_model.signature_def_utils.predict_signature_def(inputs, outputs)
    }
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map=serving_signatures,
                                         assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
                                         clear_devices=True)
    builder.save()

    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error,)


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
