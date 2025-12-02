#!/bin/bash

#cd ../..

# custom config
DATA="/data0/lvsl/DATA"

DATASET=$1
SEED=$2
DEVICE=$3
INIT_SEED=$4
TRAINER=BITP

CFG=vit_b16_c2_ep5_ctx2_dp9TT_warmup5_BZ4_ex
SHOTS=16
LOADEP=5
SUB=new


COMMON_DIR=${DATASET}/${CFG}_${INIT_SEED}/seed${SEED}
MODEL_DIR=output/base2new/train_base_SEED/${COMMON_DIR}
DIR=output/base2new/test_${SUB}_SEED/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

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
    --eval-only \
    --best \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    MODEL.INIT_SEED ${INIT_SEED}

else
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
    --best  \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    MODEL.INIT_SEED ${INIT_SEED}
fi