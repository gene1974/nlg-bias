#!/usr/bin/env bash

# echo "Model type: ${1}"
# echo "No ext sample name: ${2}"

# # Modify params.
# export DATA_DIR=$(dirname ${2})
# export TEST_BASE=$(basename ${2})

# Fixed params.
export MAX_LENGTH=128
# export TEST_FILE=${TEST_BASE}.tsv.XYZ

export BERT_MODEL_PRETRAINED_CN=./models/bert-base-chinese
# export BERT_MODEL_PRETRAINED_CN=./models/bert-base-uncased
# export BERT_MODEL_OUTPUT_CN=/data/pretrained/bert-base-chinese
export OUTPUT_DIR=./models/regard_cn
export DATA_DIR=./data/regard-chinese
export TEST_DIR=data/generated_samples_zh
export TEST_FILE=small_gpt2_generated_samples_manual.tsv

echo "Training chinese model"
python scripts/run_classifier.py --data_dir ${DATA_DIR} \
--model_type bert \
--model_name_or_path ${BERT_MODEL_PRETRAINED_CN} \
--output_dir ${OUTPUT_DIR} \
--max_seq_length  ${MAX_LENGTH} \
--do_train \
--do_predict \
--overwrite_output_dir \
--num_train_epochs 10 \
--do_lower_case \
--overwrite_cache \
--per_gpu_eval_batch_size 32 \
--model_version 2


# echo "Label with Chinese model"
# python scripts/run_classifier.py --data_dir ${TEST_DIR} \
# --model_type bert \
# --model_name_or_path ${OUTPUT_DIR} \
# --output_dir ${OUTPUT_DIR} \
# --max_seq_length  ${MAX_LENGTH} \
# --do_predict \
# --test_file ${TEST_FILE} \
# --do_lower_case \
# --overwrite_cache \
# --per_gpu_eval_batch_size 32 \
# --model_version 2
# python scripts/ensemble.py --data_dir ${ENSEMBLE_DIR} --output_prefix ${OUTPUT_PREFIX} --file_with_demographics ${DATA_DIR}/${TEST_BASE}.tsv
