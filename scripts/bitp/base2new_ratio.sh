#!/bin/bash

#cd ../..

# custom config
DATA="/data0/lvsl/DATA"
TRAINER=BITP

DATASET=$1
SEED=$2
DEVICE=$3
RATIO=$4


CFG=vit_b16_c2_ep20_ctx1_dp9TT_warmup2_BZ2_V3
SHOTS=16
LOADEP=15
SUB=new

#/home/lvsl/Code/BAITP/output/base2new/train_base/caltech101/vit_b16_c2_ep5_batch4_4/VPT/seed1/prompt_learner\
# /home/lvsl/Code/BITP/output/BEST_MODEL/train_base/Joint2
COMMON_DIR=Joint2/${DATASET}/seed${SEED}
MODEL_DIR=output/BEST_MODEL/train_base/${COMMON_DIR}
DIR=output/RATIO/Joint2/${DATASET}/RATIO_${RATIO}

echo "Evaluating model"
echo "Calculating the different ratio of data"
echo "Runing the first phase job and save the output to ${DIR}"


echo "Current ratio is ${RATIO}"
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
DATASET.SUBSAMPLE_RATIO ${RATIO}
#echo "Finish the first phase job"
#kill -9 $(ps -ef | grep "base2new_ratio1" | grep -v grep | awk '{print $2}')