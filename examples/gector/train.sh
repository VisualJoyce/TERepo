#!/usr/bin/env bash
RUN_FILE=$(readlink -f "$0")
PROJECT_DIR=$(dirname "$RUN_FILE")
WORK_DIR=$(dirname $(dirname "$PROJECT_DIR"))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"

MODEL_CLS=$1
MODEL_NAME=$2
GPU_IDS=$3
TASK=$4
PARA=$5
#  PARA="--focal_gamma 3 "

if [[ $MODEL_CLS == gector* ]]; then
  MODEL_TYPE=gector
elif [[ $MODEL_CLS == igector* ]]; then
  MODEL_TYPE=igector
else
  echo "$MODEL_NAME is not supported"
  exit
fi

MAX_STEPS=200000
STEPS=1000
LEARNING_RATE=1e-5
PYTHON_BIN=$(which python)
ANACONDA_PATH=${PYTHON_BIN%%/bin/python}
NUM_GPU=$((1+$(echo "$GPU_IDS"| tr -d -c ',' | wc -m)))
PORT=$(echo "$GPU_IDS" | python -c "import sys;print(sum(map(int,sys.stdin.read().split(','))))")
MASTER_PORT=$((23456+PORT))

if [ "$TASK" == "mucgec" ]; then
  LANG=zh
  TRAIN_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/mucgec/train
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/mucgec/valid
  EVAL_GOLD_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/mucgec/MuCGEC_dev.txt
  EVALUATOR_SUBNAMES=cherrant
  MAX_STEPS=40000
  LEARNING_RATE=1e-5
elif [ "$TASK" == "fcgec" ]; then
  LANG=zh
  TRAIN_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/fcgec/train
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/fcgec/valid
  EVAL_GOLD_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/fcgec/FCGEC_valid.txt
  EVALUATOR_SUBNAMES=cherrant
  MAX_STEPS=10000
  STEPS=100
  LEARNING_RATE=1e-5
elif [ "$TASK" == "mcscset" ]; then
  LANG=zh
  TRAIN_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/mcscset/train
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/mcscset/valid
  EVAL_GOLD_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/mcscset/valid_gold.txt
  EVALUATOR_SUBNAMES=cherrant
  MAX_STEPS=20000
  LEARNING_RATE=5e-5
  STEPS=100
elif [ "$TASK" == "bea2019" ]; then
  LANG=en
  TRAIN_FILES="${WORK_DIR}"/data/annotations/text_editing/en/synthetic/train:"${WORK_DIR}"/data/annotations/text_editing/en/fce/train:"${WORK_DIR}"/data/annotations/text_editing/en/lang8/train:"${WORK_DIR}"/data/annotations/text_editing/en/nucle/train:"${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/train
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/en/fce/dev@"${WORK_DIR}"/data/annotations/text_editing/en/fce/test@"${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/dev@"${WORK_DIR}"/data/annotations/text_editing/en/conll2014/test
  EVALUATOR_SUBNAMES=m2scorer
  MAX_STEPS=300000
  LEARNING_RATE=5e-5
elif [ "$TASK" == "wi+locness" ]; then
  LANG=en
  TRAIN_FILES="${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/train
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/en/fce/dev@"${WORK_DIR}"/data/annotations/text_editing/en/fce/test@"${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/dev
  EVALUATOR_SUBNAMES=errant
  MAX_STEPS=10000
  LEARNING_RATE=1e-6
elif [ "$TASK" == "nucle+wi+locness" ]; then
  LANG=en
  TRAIN_FILES="${WORK_DIR}"/data/annotations/text_editing/en/nucle/train:"${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/train
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/en/fce/dev@"${WORK_DIR}"/data/annotations/text_editing/en/fce/test@"${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/dev
  EVALUATOR_SUBNAMES=errant
  MAX_STEPS=10000
  LEARNING_RATE=1e-6
else
  exit
fi

STEPS=$((MAX_STEPS/100))
BATCH_SIZE=128
MODEL_DIR="${WORK_DIR}"/data/output/text_editing/$LANG/gec/"${MODEL_CLS}"-"${MODEL_NAME}"-"${TASK}"/${UUID}

GPU_TYPE=$(nvidia-smi -q | grep "Product Name" | head -n 1 | awk -F':' '{print $2}' | xargs echo -n)
if [ "$GPU_TYPE" == "NVIDIA GeForce GTX TITAN X" ]; then
  PER_DEVICE_TRAIN_BATCH_SIZE=16
elif [ "$GPU_TYPE" == "NVIDIA TITAN RTX" ]; then
  PER_DEVICE_TRAIN_BATCH_SIZE=32
else
  PER_DEVICE_TRAIN_BATCH_SIZE=32
fi

GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE/PER_DEVICE_TRAIN_BATCH_SIZE/NUM_GPU))

mkdir -p "${MODEL_DIR}"/logs
log_file="${MODEL_DIR}"/logs/train.txt
exec &> >(tee -a "$log_file")

set -x

#HOST_GPU_NUM=1
export LD_LIBRARY_PATH=$ANACONDA_PATH/lib
#CUDA_VISIBLE_DEVICES=$GPU_IDS PYTHONPATH="${WORK_DIR}"/src python3 \
CUDA_VISIBLE_DEVICES=$GPU_IDS PYTHONPATH="${WORK_DIR}"/src torchrun --nproc_per_node="$NUM_GPU" --master_port=$MASTER_PORT \
  "${PROJECT_DIR}"/train.py \
  --run_name "$UUID" \
  --train_files "${TRAIN_FILES}" \
  --train_loader_names tagging --train_loader_subnames $MODEL_TYPE \
  --eval_files "${EVAL_FILES}" \
  --eval_loader_names tagging --eval_loader_subnames $MODEL_TYPE \
  --evaluator_subnames $EVALUATOR_SUBNAMES \
  --eval_gold_file "$EVAL_GOLD_FILE" \
  --model_name_or_path "${MODEL_NAME}" $PARA \
  --model_cls "${MODEL_CLS}" \
  --fp16 --do_train --do_eval \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay 0.01 --warmup_steps $STEPS --save_steps $STEPS --logging_steps $STEPS --block_size 128 \
  --output_dir "${MODEL_DIR}"/models \
  --logging_dir "${MODEL_DIR}"/logs \
  --log_on_each_node false --max_steps $MAX_STEPS
