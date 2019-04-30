# coding=utf-8
""" Problem definition for translation from Up to Down."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

#from tensor2tensor.data_generators.wmt import WMTProblem
from tensor2tensor.data_generators.translate import TranslateProblem
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer


SRC_LAN = 'up'
TGT_LAN = 'down'
SRC_TRAIN_DATA = 'Your up train file'
TGT_TRAIN_DATA = 'Your down train file'
SRC_DEV_DATA = 'Your up dev file'
TGT_DEV_DATA = 'Your down dev file'
MERGE_VOCAB = 'merge.tok.vocab.clean'
VOCAB_SIZE = 7033
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


def bi_vocabs_token2id_generator(source_path, target_path, token_vocab, eos=None):
    """Generator for sequence-to-sequence tasks that uses tokens.

    This generator assumes the files at source_path and target_path have
    the same number of lines and yields dictionaries of "inputs" and "targets"
    where inputs are token ids from the " "-split source (and target, resp.) lines
    converted to integers using the token_map.

    Args:
      source_path: path to the file with source sentences.
      target_path: path to the file with target sentences.
      source_token_vocab: text_encoder.TextEncoder object.
      target_token_vocab: text_encoder.TextEncoder object.
      eos: integer to append at the end of each sequence (default: None).

    Yields:
      A dictionary {"inputs": source-line, "targets": target-line} where
      the lines are integer lists converted from tokens in the file lines.
    """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                source_ints = token_vocab.encode(source.strip()) + eos_list
                target_ints = token_vocab.encode(target.strip()) + eos_list
                yield {"inputs": source_ints, "targets": target_ints}
                source, target = source_file.readline(), target_file.readline()


@registry.register_problem
class TranslateUp2down(TranslateProblem):
    """Problem spec for Up to Down translation."""


    @property
    def vocab_size(self):
        return VOCAB_SIZE # subtract for compensation

    @property
    def num_shards(self):
        return 1

    @property
    def vocab_name(self):
        return MERGE_VOCAB

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK
    
    
    # Pre-process two vocabularies and build a generator.
    def generator(self, data_dir, tmp_dir, train):
        # Load vocabulary.
        tf.logging.info("Loading and processing vocabulary for %s from:" % ("training" if train else "validation"))
        print('    ' + _VOCAB_FILES[0] + ' ... ', end='')
        sys.stdout.flush()
        with open(_VOCAB_FILES[0], 'r', encoding='utf-8') as f:
            vocab_list = f.read().splitlines()
        print('Done')
        
        # Truncate the vocabulary depending on the given size (strip the reserved tokens).
        vocab_list = vocab_list[3:]
        
        # Insert the <UNK>.
        vocab_list.insert(0, "<UNK>")
        
        # Auto-insert the reserved tokens as: <pad>=0 <EOS>=1 and <UNK>=2.
        vocab = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_list,
                                                     replace_oov="<UNK>", num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)
        
        # Select the path: train or dev (small train).
        datapath = _TRAIN_DATASETS if train else _DEV_DATASETS
        
        # Build a generator.
        return bi_vocabs_token2id_generator(datapath[0], datapath[1], vocab, text_encoder.EOS_ID)
    
    
    # Build bi-vocabs feature encoders for decoding.
    def feature_encoders(self, data_dir):
        # Load vocabulary.
        tf.logging.info("Loading and processing vocabulary from: %s" % _VOCAB_FILES[0])
        with open(_VOCAB_FILES[0], 'r', encoding='utf-8') as f:
            vocab_list = f.read().splitlines()
        tf.logging.info("Done")
        
        # Truncate the vocabulary depending on the given size (strip the reserved tokens).
        vocab_list = vocab_list[3:]
        
        # Insert the <UNK>.
        vocab_list.insert(0, "<UNK>")
        
        # Auto-insert the reserved tokens as: <pad>=0 <EOS>=1 and <UNK>=2.
        encoder = text_encoder.TokenTextEncoder(vocab_filename=None, vocab_list=vocab_list,
                                                       replace_oov="<UNK>", 
                                                        num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS)
        
        return {"inputs": encoder, "targets": encoder}



