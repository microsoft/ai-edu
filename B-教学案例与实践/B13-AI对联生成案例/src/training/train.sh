#!/bin/bash

HOME_DIR=$(cd `dirname $0`; pwd)
echo $HOME_DIR

# clone and checkout tensor2tensor library to v1.2.9
cd $HOME_DIR
git clone https://github.com/tensorflow/tensor2tensor.git
cd tensor2tensor
git checkout v1.2.9
cd ..
mv tensor2tensor tensor2tensor-1.2.9

# set environment
CODE_DIR=${HOME_DIR}/tensor2tensor-1.2.9
export PYTHONPATH=${CODE_DIR}:${PYTHONPATH}
binFile=${CODE_DIR}/tensor2tensor/bin

TRAIN_DIR=${HOME_DIR}/output
LOG_DIR=${TRAIN_DIR}

DATA_DIR=${HOME_DIR}/data
USR_DIR=${DATA_DIR}


PROBLEM=translate_up2down
MODEL=transformer
HPARAMS_SET=transformer_small
#HPARAMS_SET=transformer_base
mkdir -p ${TRAIN_DIR}

#install python packages
python -m pip install -r pip_requirements.txt

# generate data
python ${binFile}/t2t-datagen \
  --t2t_usr_dir=${USR_DIR} \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM}

# train model
setting=default

python ${binFile}/t2t-trainer \
--t2t_usr_dir=${USR_DIR} \
--data_dir=${DATA_DIR} \
--problems=${PROBLEM} \
--model=${MODEL} \
--hparams_set=${HPARAMS_SET} \
--output_dir=${TRAIN_DIR} \
--keep_checkpoint_max=1000 \
--worker_gpu=1 \
--train_steps=200000 \
--save_checkpoints_secs=1800 \
--schedule=train \
--worker_gpu_memory_fraction=0.95 \
--hparams="batch_size=1024" 2>&1 | tee -a ${LOG_DIR}/train_${setting}.log
