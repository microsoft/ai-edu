# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "", "Training directory to load from.")
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file", None,
                    "Path prefix to inference output file")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("t2t_usr_dir", "",
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-decoder.")
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "train_and_evaluate",
                    "Must be train_and_evaluate for decoding.")

home = os.path.join(os.getenv("HOME"), "code", "test_model")
FLAGS.t2t_usr_dir = home + '/usr_dir'
FLAGS.data_dir = home + '/data_dir'
FLAGS.output_dir =  home + '/model'
FLAGS.problems = 'translate_up2down'
FLAGS.model = 'transformer'
FLAGS.hparams = 'batch_size=1024'
FLAGS.hparams_set = 'transformer_small'
FLAGS.decode_hparams = "beam_size=4,alpha=0.6,batch_size=32"
FLAGS.input_sentence = "花 开 花 落"
FLAGS.decode_shards = 1
FLAGS.worker_id = 1
FLAGS.decode_from_file = True
FLAGS.decode_to_file = True

class up2down_class:
  def __init__(self):

    tf.logging.set_verbosity(tf.logging.INFO)
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    trainer_utils.log_registry()
    #trainer_utils.validate_flags()
    self.data_dir = os.path.expanduser(FLAGS.data_dir)
    self.output_dir = os.path.expanduser(FLAGS.output_dir)

    self.hparams = trainer_utils.create_hparams(
      FLAGS.hparams_set, self.data_dir, passed_hparams=FLAGS.hparams)
    trainer_utils.add_problem_hparams(self.hparams, FLAGS.problems)
    self.estimator, _ = trainer_utils.create_experiment_components(
      data_dir=self.data_dir,
      model_name=FLAGS.model,
      hparams=self.hparams,
      run_config=trainer_utils.create_run_config(self.output_dir))

    self.decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
    self.decode_hp.add_hparam("shards", FLAGS.decode_shards)
    self.decode_hp.add_hparam("shard_id", FLAGS.worker_id)
    output_sentence = decoding.decode_from_file(self.estimator, FLAGS.decode_from_file, self.decode_hp,
                                FLAGS.decode_to_file, input_sentence=FLAGS.input_sentence)

  def get_next(self, input_sentence):
    input_sentence_modify = ""
    for chr in input_sentence.strip():
      input_sentence_modify += chr
      input_sentence_modify += " "
    input_sentence_modify = input_sentence_modify[:-1]
    output_sentence_modify = decoding.decode_from_file(self.estimator, FLAGS.decode_from_file, self.decode_hp,
                                FLAGS.decode_to_file, input_sentence=input_sentence)
    output_sentence = "".join(output_sentence_modify.split())
    return output_sentence
