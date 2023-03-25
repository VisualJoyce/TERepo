#!/usr/bin/env bash
RUN_FILE=$(readlink -f "$0")
PROJECT_DIR=$(dirname "$RUN_FILE")
WORK_DIR=$(dirname $(dirname "$PROJECT_DIR"))
echo "${WORK_DIR}"

MODEL_CLS=$1
MODEL_NAME=$2
MODEL_UUID=$3
MODEL_CKPT=$4
MODEL_TASK=$5
GPU_ID=$6
EVAL_TASK=$7
ITERATION_COUNT=$8

MIN_ERROR_PROBABILITY=$9
CONFIDENCE_BIAS=${10}

if [ "$MODEL_TASK" == "mucgec" ]; then
  LANG=zh
  if [ -z $MIN_ERROR_PROBABILITY ]; then
    MIN_ERROR_PROBABILITY=0
  fi
  if [ -z $CONFIDENCE_BIAS ]; then
    CONFIDENCE_BIAS=0
  fi
elif [ "$MODEL_TASK" == "fcgec" ]; then
  LANG=zh
  if [ -z $MIN_ERROR_PROBABILITY ]; then
    MIN_ERROR_PROBABILITY=0
  fi
  if [ -z $CONFIDENCE_BIAS ]; then
    CONFIDENCE_BIAS=0
  fi
elif [ "$MODEL_TASK" == "mcscset" ]; then
  LANG=zh
  if [ -z $MIN_ERROR_PROBABILITY ]; then
    MIN_ERROR_PROBABILITY=0
  fi
  if [ -z $CONFIDENCE_BIAS ]; then
    CONFIDENCE_BIAS=0
  fi
elif [ "$MODEL_TASK" == "bea2019" ]; then
  LANG=en
  if [ -z $MIN_ERROR_PROBABILITY ]; then
    MIN_ERROR_PROBABILITY=0.41
  fi
  if [ -z $CONFIDENCE_BIAS ]; then
    CONFIDENCE_BIAS=0.1
  fi
elif [ "$MODEL_TASK" == "wi+locness" ]; then
  LANG=en
  if [ -z $MIN_ERROR_PROBABILITY ]; then
    MIN_ERROR_PROBABILITY=0.41
  fi
  if [ -z $CONFIDENCE_BIAS ]; then
    CONFIDENCE_BIAS=0.1
  fi
elif [ "$MODEL_TASK" == "nucle+wi+locness" ]; then
  LANG=en
  if [ -z $MIN_ERROR_PROBABILITY ]; then
    MIN_ERROR_PROBABILITY=0.41
  fi
  if [ -z $CONFIDENCE_BIAS ]; then
    CONFIDENCE_BIAS=0.1
  fi
else
  exit
fi

CKPT_DIR="${WORK_DIR}"/data/output/text_editing/$LANG/gec/"${MODEL_CLS}"-"${MODEL_NAME}"-"${MODEL_TASK}"/"${MODEL_UUID}"/models/"${MODEL_CKPT}"

if [ "$EVAL_TASK" == "mucgec" ]; then
  INPUT_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/mucgec/MuCGEC_test.txt
  OUTPUT_FILE="${CKPT_DIR}"/MuCGEC_test.txt.$ITERATION_COUNT.$MIN_ERROR_PROBABILITY.$CONFIDENCE_BIAS
elif [ "$EVAL_TASK" == "fcgec" ]; then
  INPUT_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/fcgec/FCGEC_test.json
  OUTPUT_FILE="${CKPT_DIR}"/predict.json.$ITERATION_COUNT.$MIN_ERROR_PROBABILITY.$CONFIDENCE_BIAS
elif [ "$EVAL_TASK" == "mcscset" ]; then
  INPUT_FILE="${WORK_DIR}"/data/annotations/text_editing/zh/mcscset/test_gold.txt
  OUTPUT_FILE="${CKPT_DIR}"/MCSCSet_test.txt.$ITERATION_COUNT.$MIN_ERROR_PROBABILITY.$CONFIDENCE_BIAS
elif [ "$EVAL_TASK" == "bea2019" ]; then
  INPUT_FILE="${WORK_DIR}"/data/annotations/text_editing/en/bea2019
  OUTPUT_FILE="${CKPT_DIR}"/ABCN.test.bea19.txt.$ITERATION_COUNT.$MIN_ERROR_PROBABILITY.$CONFIDENCE_BIAS
else
  exit
fi

## declare an array variable
declare -a host_array=("3090" "titan-85" "titan-86")

# get length of an array
array_length=${#host_array[@]}

if [ "$MODEL_CKPT" == "ckpt0" ]; then
  mkdir -p "$CKPT_DIR"
  BEST_PT=
else
  BEST_PT="--best_pt ${CKPT_DIR}/pytorch_model.bin"
  if [ ! -f "${CKPT_DIR}"/pytorch_model.bin ]; then
    # use for loop to read all values and indexes
    for ((i = 0; i < ${array_length}; i++)); do
      SERVER=${host_array[$i]}
      echo "index: $i, value: $SERVER"
      rsync -av --exclude='*.bin' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.tar' "$SERVER":Data/output /home/tanminghuan/Data
      scp "${SERVER}":"${CKPT_DIR}"/pytorch_model.bin "${CKPT_DIR}"
    done
  fi
fi


log_file="${CKPT_DIR}"/logs.$EVAL_TASK."$ITERATION_COUNT".$MIN_ERROR_PROBABILITY.$CONFIDENCE_BIAS.txt
exec &> >(tee -a "$log_file")

set -x

export LD_LIBRARY_PATH=/home/tanminghuan/anaconda3/lib
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="${WORK_DIR}"/src python3.9 "${PROJECT_DIR}"/predict.py \
  --model_name_or_path "${MODEL_NAME}" ${BEST_PT} \
  --predictor tagging --predictor_subname gector --task "$EVAL_TASK" \
  --model_cls "${MODEL_CLS}" \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --iteration_count "$ITERATION_COUNT" \
  --min_error_probability ${MIN_ERROR_PROBABILITY} \
  --confidence_bias ${CONFIDENCE_BIAS}

if [ "$EVAL_TASK" == "mcscset" ]; then
  MCSCSET_ALL_M2=${INPUT_FILE}.char   # ChERRANT, char-based
  if [ ! -f $MCSCSET_ALL_M2 ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="${WORK_DIR}"/src python3.9 "${WORK_DIR}"/src/ChERRANT/parallel_to_m2.py -f $INPUT_FILE -o $MCSCSET_ALL_M2 -g char -s all
  fi
  OUTPUT_FILE_M2=$OUTPUT_FILE.char
  CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="${WORK_DIR}"/src python3.9 "${WORK_DIR}"/src/ChERRANT/parallel_to_m2.py -f $OUTPUT_FILE -o $OUTPUT_FILE_M2 -g char -s all
  python3.9 "${WORK_DIR}"/src/ChERRANT/compare_m2_for_evaluation.py -hyp $OUTPUT_FILE_M2 -ref $MCSCSET_ALL_M2
fi

echo "$OUTPUT_FILE"
