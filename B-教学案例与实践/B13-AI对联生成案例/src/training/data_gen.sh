HOME_DIR=$(cd `dirname $0`; pwd)
CODE_DIR=${HOME_DIR}/tensor2tensor
export PYTHONPATH=${CODE_DIR}:${PYTHONPATH}
binFile=${CODE_DIR}/tensor2tensor/bin

PROBLEM=translate_up2down
MODEL=transformer
HPARAMS_SET=transformer_small

DATA_DIR=${HOME_DIR}/data

mkdir -p $DATA_DIR

# Generate data
python ${binFile}/t2t-datagen \
  --t2t_usr_dir=$DATA_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM
