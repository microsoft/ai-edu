PROBLEM=translate_up2down
MODEL=transformer
HPARAMS_SET=transformer_small

HOME_DIR=$(cd `dirname $0`; pwd)
CODE_DIR=$HOME_DIR/tensor2tensor-1.2.9
DATA_DIR=$HOME_DIR/data

USR_LIB_DIR=$HOME_DIR/usr_dir

mosesdecoder=${HOME_DIR}/mosesdecoder

TRAIN_DIR=${HOME_DIR}/train
TEST_DIR=$HOME_DIR/test
LOG_DIR=$HOME_DIR/log
export PYTHONPATH=$CODE_DIR:$PYTHONPATH

mkdir -p $TEST_DIR

BEAM_SIZE=4
ALPHA=0.6
BATCH_SIZE=32

START_POINT=1
First_lan_file=
Second_lan_file=
#EXTRA=supervise1


while true; do
  ids=$(ls ${TRAIN_DIR} | grep "model\.ckpt-[0-9]*.index" | grep -o "[0-9]*")
  echo "All Ids:" ${ids}
 
  for i in ${ids}; do
    if [ $i -gt $START_POINT ]; then
      TEST_RESULT_DIR=${TEST_DIR}/$i-BEAM${BEAM_SIZE}-ALPHA${ALPHA}  # in TMP_DIR
      if test -s $TEST_RESULT_DIR/BLEU.txt; then
        echo $i, "already tested"
      else
        echo "testing" $i
        rm -rf $TEST_RESULT_DIR
        mkdir -p $TEST_RESULT_DIR
        cp $TRAIN_DIR/model.ckpt-${i}.* $TEST_RESULT_DIR
        touch $TEST_RESULT_DIR/checkpoint
        echo model_checkpoint_path: \"model.ckpt-${i}\" >> $TEST_RESULT_DIR/checkpoint
        echo all_model_checkpoint_paths: \"model.ckpt-${i}\" >> $TEST_RESULT_DIR/checkpoint
        
    #the test result has finished bpe/token
        cp $DATA_DIR/$First_lan_file $TEST_RESULT_DIR
        cp $DATA_DIR/$Second_lan_file $TEST_RESULT_DIR

        
        #Fir -> Sec

        python $CODE_DIR/tensor2tensor/bin/t2t-decoder \
          --t2t_usr_dir=$USR_LIB_DIR \
          --data_dir=$DATA_DIR \
          --problems=$PROBLEM \
          --model=$MODEL \
          --hparams_set=$HPARAMS_SET \
          --hparams="batch_size=1024" \
          --output_dir=$TEST_RESULT_DIR \
          --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,batch_size=$BATCH_SIZE" \
          --worker_gpu=1 \
          --decode_from_file=$TEST_RESULT_DIR/$First_lan_file \
          --decode_to_file=$TEST_RESULT_DIR/$First_lan_file.${i} \
          2>&1 \
        | tee $LOG_DIR/test-$i-BEAM${BEAM_SIZE}-ALPHA${ALPHA}-ENWAR.log.txt

        mv $TEST_RESULT_DIR/$First_lan_file.${i}*.decodes $TEST_RESULT_DIR/$First_lan_file.${i}.decodes
    echo "EN -> WAR:" >> $TEST_RESULT_DIR/BLEU.txt
        $mosesdecoder/scripts/generic/multi-bleu.perl \
        $TEST_RESULT_DIR/$Second_lan_file < $TEST_RESULT_DIR/$First_lan_file.${i}.decodes \
        | tee -a $TEST_RESULT_DIR/BLEU.txt
        
        rm -r $TEST_RESULT_DIR/model.ckpt-*
    echo "$id:"
    cat $TEST_RESULT_DIR/BLEU.txt
      fi
    else
      echo "pass" $i
    fi
  done
  cat $TEST_DIR/ALL_BLEU.txt
  break
done





