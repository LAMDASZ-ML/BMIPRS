#!/bin/bash

#cd ../..

# custom config
DATA="/data0/lvsl/DATA"
TRAINER=BITP

DATASET=$1
SEED=$2
DEVICE=$3
INIT_SEED=$4

CFG=vit_b16_c2_ep5_ctx2_dp9TT_warmup5_BZ4_ex
SHOTS=16


DIR=output/base2new/train_base_SEED/${DATASET}/${CFG}_${INIT_SEED}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --device ${DEVICE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    MODEL.INIT_SEED ${INIT_SEED}
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --device ${DEVICE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    MODEL.INIT_SEED ${INIT_SEED}
fi