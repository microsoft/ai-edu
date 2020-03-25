# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from functools import reduce
# Dependency imports
from tensor2tensor.bin import t2t_trainer # important
from tensor2tensor.utils import usr_dir
from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import hparam
import tensorflow as tf

# Load config
with open('./config.json','r') as f:
    config = json.load(f)

flags = tf.flags
FLAGS = flags.FLAGS
FLAGS.t2t_usr_dir = config['t2t_usr_dir']
FLAGS.problem = config['problem']
FLAGS.model = config['model_name']
server_address = config['server_address']

class up2down_class:

    def __init__(self,FLAGS,server_address='127.0.0.1:9000'):
        print('Initializing up2down_class.......')
        self.FLAGS = FLAGS
        self.server_address = server_address
        tf.logging.set_verbosity(tf.logging.ERROR)

        usr_dir.import_usr_dir(self.FLAGS.t2t_usr_dir)
        # hparams: not important but necessary, an assertion error will be raised without hparams.
        self.hparams = hparam.HParams(data_dir=os.path.expanduser(self.FLAGS.t2t_usr_dir))
        # problem
        self.problem = registry.problem(self.FLAGS.problem)
        self.problem.get_hparams(self.hparams)
        # model request server
        self.request_fn = self.make_request_fn(self.FLAGS.model, self.server_address)


    def make_request_fn(self, server_name, server_address):
        """Returns a request function."""
        request_fn = serving_utils.make_grpc_request_fn(
            servable_name=server_name,
            server=server_address,
            timeout_secs=10)

        return request_fn

    def get_down_couplet(self, input_sentence_raw_list):

        input_sentence_list = self.format_input(input_sentence_raw_list)
        # do inference
        raw_outputs = serving_utils.predict(input_sentence_list, self.problem, self.request_fn)
        
        outputs = self.format_output(raw_outputs)

        return outputs

    def format_input(self, input_sentence_raw_list):
        input_sentence_list = []
        for input_sentence in input_sentence_raw_list:
            input_sentence_modify = ""
            for chr in input_sentence.strip():
                input_sentence_modify += chr
                input_sentence_modify += " "
            input_sentence_modify = input_sentence_modify[:-1]
            input_sentence_list.append(input_sentence_modify)
        print("input sentences: " + str(input_sentence_list))
        return input_sentence_list

    def format_output(self, raw_outputs):
        outputs = []
        for raw_output in raw_outputs:
            out = raw_output[0].replace(' ', '').replace('<EOS>', '').replace('<pad>', '')
            outputs.append(out)
        return outputs

### Export model ###
up2down = up2down_class(FLAGS,server_address) # inference model