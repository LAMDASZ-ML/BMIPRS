#!/bin/bash

#cd ../..

# custom config
DATA="/mnt/hdd/DATA"

DATASET=$1
SEED=$2
DEVICE=$3
TRAINER=BMIPKL

# CFG=BMIP_ctx2_dp9_lr2_seed2020
INIT_SEED=$(printf "%.1f" "$4")  # 保留1位小数，例如 $4=3 时，结果为 3.0
CFG=$5
SHOTS=16
LOADEP=12
SUB=new


COMMON_DIR=${CFG}_${INIT_SEED}/${DATASET}/seed${SEED}
MODEL_DIR=./output/base2new/train_base/${COMMON_DIR}
DIR=./output/base2new/test_${SUB}/${COMMON_DIR}
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
    --best \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.BMIPKL.KL_LOSS_WEIGHT ${INIT_SEED}
    # MODEL.INIT_SEED ${INIT_SEED}

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
    --best \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.BMIPKL.KL_LOSS_WEIGHT ${INIT_SEED}
    # MODEL.INIT_SEED ${INIT_SEED}
fi
