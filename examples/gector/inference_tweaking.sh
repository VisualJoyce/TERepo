#!/usr/bin/env bash
WORK_DIR=$(dirname $(dirname $(readlink -f $0)))
echo "${WORK_DIR}"

MODEL_CLS=$1
MODEL_NAME=$2
MODEL_UUID=$3
MODEL_CKPT=$4
MODEL_TASK=$5
GPU_ID=$6

if [ "$MODEL_TASK" == "mucgec" ]; then
  LANG=zh
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/mucgec/valid
  EVAL_GOLD_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/mucgec/MuCGEC_dev.txt
  EVALUATOR_SUBNAMES=cherrant
elif [ "$MODEL_TASK" == "fcgec" ]; then
  LANG=zh
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/fcgec/valid
  EVAL_GOLD_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/fcgec/FCGEC_valid.txt
  EVALUATOR_SUBNAMES=cherrant
elif [ "$MODEL_TASK" == "mcscset" ]; then
  LANG=zh
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/zh/mcscset/valid
  EVAL_GOLD_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/mcscset/valid_gold.txt
  EVALUATOR_SUBNAMES=cherrant
elif [ "$MODEL_TASK" == "bea2019" ]; then
  LANG=en
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/en/fce/dev@"${WORK_DIR}"/data/annotations/text_editing/en/fce/test@"${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/dev
  EVALUATOR_SUBNAMES=errant
elif [ "$MODEL_TASK" == "wi+locness" ]; then
  LANG=en
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/dev
  EVALUATOR_SUBNAMES=errant
elif [ "$MODEL_TASK" == "nucle+wi+locness" ]; then
  LANG=en
  EVAL_FILES="${WORK_DIR}"/data/annotations/text_editing/en/wi+locness/dev
  EVALUATOR_SUBNAMES=errant
else
  exit
fi

CKPT_DIR="${WORK_DIR}"/data/output/text_editing/$LANG/gec/"${MODEL_CLS}"-"${MODEL_NAME}"-"${MODEL_TASK}"/"${MODEL_UUID}"/models/"${MODEL_CKPT}"

if [ "$MODEL_CKPT" == "ckpt0" ]; then
  mkdir -p "$CKPT_DIR"
  BEST_PT=
else
  BEST_PT="--best_pt ${CKPT_DIR}/pytorch_model.bin"
fi

BATCH_SIZE=128

GPU_TYPE=$(nvidia-smi -q | grep "Product Name" | head -n 1 | awk -F':' '{print $2}' | xargs echo -n)
if [ "$GPU_TYPE" == "NVIDIA GeForce GTX TITAN X" ]; then
  PER_DEVICE_EVAL_BATCH_SIZE=$BATCH_SIZE
elif [ "$GPU_TYPE" == "NVIDIA TITAN RTX" ]; then
  PER_DEVICE_EVAL_BATCH_SIZE=512
else
  PER_DEVICE_EVAL_BATCH_SIZE=512
fi

log_file="${CKPT_DIR}"/logs.inference_tweaking.txt
exec &> >(tee -a "$log_file")

HOST_GPU_NUM=1
export LD_LIBRARY_PATH=/home/tanminghuan/anaconda3/lib
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="${WORK_DIR}"/src torchrun --nproc_per_node="$HOST_GPU_NUM" --master_port=2345"${GPU_ID}" \
  "${WORK_DIR}"/examples/inference_tweaking.py \
  --eval_files "${EVAL_FILES}" \
  --eval_loader_names tagging_inference_tweaking --eval_loader_subnames gector \
  --evaluator_subnames $EVALUATOR_SUBNAMES \
  --eval_gold_file "$EVAL_GOLD_FILE" \
  --model_name_or_path "${WORK_DIR}"/data/pretrained_models/text_editing/"${MODEL_NAME}" $BEST_PT \
  --model_cls "${MODEL_CLS}" \
  --fp16 --do_eval \
  --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
  --weight_decay 0.01 --warmup_steps 1000 --save_steps 1000 --logging_steps 1000 --block_size 128 \
  --output_dir "${CKPT_DIR}" \
  --logging_dir "${CKPT_DIR}"