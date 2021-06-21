#!/bin/bash

# Before you inference, please make sure your directory structure is as follow:
#
# usr_dir \
#     __init__.py
#     merge_vocab.py
# data_dir \
#     train.txt.up.clean
#     train.txt.down.clean
#     merge.txt.vocab.clean
# output \
#

TRAIN_DIR=./output
DATA_DIR=./data_dir
USR_DIR=./usr_dir

DECODE_FILE=./decode_this.txt

PROBLEM=translate_up2down
MODEL=transformer
HPARAMS=transformer_small

BEAM_SIZE=4
ALPHA=0.6

poet=$1
new_chars=""
for ((i=0;i < ${#poet} ;++i))
do
new_chars="$new_chars ${poet:i:1}"
done

echo $new_chars > decode_this.txt

echo "生成中..."

t2t-decoder \
--t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=result.txt >> /dev/null 2>&1

echo $new_chars
cat result.txt