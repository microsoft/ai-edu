# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding=utf-8
""" Problem definition for translation from Up to Down."""
## version info: tensor2tensor 1.14.1 and tensorflow 1.14.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

SRC_TRAIN_DATA = 'train.txt.up.clean'
TGT_TRAIN_DATA = 'train.txt.down.clean'
SRC_DEV_DATA = 'dev.txt.up.clean'
TGT_DEV_DATA = 'dev.txt.down.clean'
MERGE_VOCAB = 'merge.txt.vocab.clean'
VOCAB_SIZE = 9122
LOCATION_OF_DATA = os.path.abspath(os.path.dirname(__file__)) + '/'

_TRAIN_DATASETS = [
    LOCATION_OF_DATA + SRC_TRAIN_DATA,
    LOCATION_OF_DATA + TGT_TRAIN_DATA
]

_DEV_DATASETS = [
    LOCATION_OF_DATA + SRC_DEV_DATA,
    LOCATION_OF_DATA + TGT_DEV_DATA
]

_VOCAB_FILES = [
    LOCATION_OF_DATA + MERGE_VOCAB
]

EOS = text_encoder.EOS_ID

@registry.register_problem
# class TranslateUp2down(text_problems.Text2TextProblem):
class TranslateUp2down(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return VOCAB_SIZE

    @property
    def vocab_name(self):
        return _VOCAB_FILES[0]

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        },{
            "split": problem.DatasetSplit.EVAL,
            "shards":1,
        }]

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        datasets = _TRAIN_DATASETS if train else _DEV_DATASETS

        vocab_list = []
        print("=======Get Vocab from ", self.vocab_name, '...', end='')
        with open(self.vocab_name, 'r', encoding='utf-8') as f:
            vocab_list = f.read().splitlines()
        print("=======Done")

        vocab = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_list, replace_oov="<UNK>", num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)

        return text_problems.text2text_generate_encoded(text_problems.text2text_txt_iterator(datasets[0], datasets[1]), vocab, vocab)


    def feature_encoders(self,data_dir):
        tf.logging.info("Loading and processing vocabulary from: %s" % _VOCAB_FILES[0])
        vocab_list = []
        with open(self.vocab_name, 'r', encoding='utf-8') as f:
            vocab_list = f.read().splitlines()
        tf.logging.info("Done")
        vocab_token = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_list, replace_oov="<UNK>", num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)
        return {"inputs": vocab_token, "targets": vocab_token}
