import os
import sys
import itertools
import logging
import re
import json
import time
import math
import tensorflow as tf
import numpy as np
import cv2

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

logging.basicConfig(level=logging.INFO)

# Override flags to make it easier to run on Philly
tf.app.flags.DEFINE_string('model_path', './image_to_vector/checkpoint', 'model dir override')
tf.app.flags.DEFINE_string('model_name', 'vgg_16', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'customized batch size')
tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS

CHANNELS = 3
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

means = [_R_MEAN, _G_MEAN, _B_MEAN]

class ModelInstance:
    def __init__(self):
        logging.info('Building model graph...')
        graph, self.images, self.features = self.build_graph()

        logging.info('Starting training session...')
        self.sess = tf.Session(graph = graph, config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=False))
    
        if tf.gfile.IsDirectory(FLAGS.model_path):
            model_path = os.path.join(FLAGS.model_path, FLAGS.model_name + '.ckpt')
        else:
            model_path = FLAGS.model_path

        self.saver.restore(self.sess, model_path)

    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            tf_global_step = slim.get_or_create_global_step()

            ####################
            # Select the model #
            ####################
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=0,
                is_training=False)

            self.eval_image_size = network_fn.default_image_size

            self.image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.model_name, is_training=False)

            ####################
            # Define the model #
            ####################
            images = tf.placeholder(tf.float32, shape=(None, self.eval_image_size, self.eval_image_size, CHANNELS))
            features, _ = network_fn(images)

            self.saver = tf.train.Saver()
            return graph, images, features

    def EvalFile(self, p_input_file, p_output_file):
            f = open(p_input_file, 'r')
            fout = open(p_output_file, 'w')

            images = np.zeros([FLAGS.batch_size, self.eval_image_size, self.eval_image_size, CHANNELS], dtype=np.float32)
            metas = []
            i = 0
            for line in f:
                imgpath = line[0:-1]
                try:
                    img = cv2.imread(imgpath)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.eval_image_size, self.eval_image_size))
                    img = img.astype(np.float32, copy=False)
                    for c in range(3):
                        img[:, :, c] = img[:, :, c] - means[c]
                except IOError:
                    continue

                images[i] = img
                metas.append(imgpath)
                i += 1

                if i == FLAGS.batch_size:
                    features = self.sess.run(self.features, feed_dict={self.images: images})
                    for it in range(i):
                        fout.write(metas[it] + '\t' + self.castvector(features[it]) + '\n')
                    metas = []
                    i = 0
            if i > 0:
                features = self.sess.run(self.features, feed_dict={self.images: images})
                for it in range(i):
                    fout.write(metas[it] + '\t' + self.castvector(features[it]) + '\n')

            f.close()
            fout.close()
        
    def Predict(self, data):
        results = []
        for q in data:
            try:
                img = cv2.imread(q)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.eval_image_size, self.eval_image_size))
                img = img.astype(np.float32, copy=False)
                for c in range(3):
                    img[:, :, c] = img[:, :, c] - means[c]
            except IOError:
                results.append(None)
                continue

            img = np.expand_dims(img, axis=0)
            features = self.sess.run(self.features, feed_dict={self.images: img})
            results.append(features[0])
        return results

    def castvector(self, vec):
        vec = np.squeeze(vec)
        res = ""
        for i in range(len(vec)):
            if res == "":
                res  = "%.5f" % vec[i]
            else:
                res += "|" + "%.5f" % vec[i]
        return res
 
