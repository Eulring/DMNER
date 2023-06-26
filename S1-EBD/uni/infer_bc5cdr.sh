#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh

REPO_PATH=./src
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=gpt
DATA_DIR=./dataset/bc5cdr
BERT_DIR=../embed/biobert-v1.1
MAX_LEN=100
MODEL_CKPT=./outputs/uni/epoch=#ID.ckpt
HPARAMS_FILE=./outputs/uni/lightning_logs/version_0/hparams.yaml


CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}


python predict2BEL.py \
-input_gold  ${DMNER_ROOT}/DATA/Datasets/BC5CDR-UNI/test.json \
-input_predict ${DATA_DIR}/predict.json \
-output_path ${DMNER_ROOT}/S2-BEM/testdata/BC5CDR-UNI/test_uni.json





DATA_DIR=./dataset/bc5cdr_dev

CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}


python predict2BEL.py \
-input_gold ${DMNER_ROOT}/DATA/Datasets/BC5CDR-UNI/validation.json \
-input_predict ${DATA_DIR}/predict.json \
-output_path ${DMNER_ROOT}/S2-BEM/testdata/BC5CDR-UNI/dev_uni.json

