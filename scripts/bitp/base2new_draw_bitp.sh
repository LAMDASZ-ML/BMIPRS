#!/bin/bash

# custom config
DATA="/data0/lvsl/DATA"

DATASET=$1
SEED=$2
TRAINER=$3
CFG=$4
SUB=$5

DEVICE=cuda:0
SHOTS=16
LOADEP=5



COMMON_DIR=${CFG}/${DATASET}/seed${SEED}
# MODEL_DIR=output/BEST_MODEL/train_base/${COMMON_DIR}
tmp=${DATASET}/base/seed${SEED}
MODEL_DIR=/mnt/hdd/lvsl/base2new/${tmp}
DIR=output/draw/test_${SUB}/${COMMON_DIR}

echo "Evaluating model"
echo "Runing the first phase job and save the output to ${DIR}"

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--device ${DEVICE} \
--draw \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}