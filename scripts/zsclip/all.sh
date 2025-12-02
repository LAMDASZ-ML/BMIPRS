#!/bin/bash

# ====== 基础配置 ======
DATA="/mnt/hdd/DATA"
TRAINER=ZeroshotCLIP
CFG=vit_b16   # rn50, rn101, vit_b32, vit_b16
SUB="all"

# ====== 数据集列表 ======
DATASETS=("aid" "mlrsnet" "optimal" "patternnet" "resisc45" "rsicb256" "whurs19")
# DATASETS=("patternnet" "resisc45" "rsicb256" "whurs19")

# ====== 逐个数据集运行 ======
for DATASET in "${DATASETS[@]}"; do
  echo "=========================================="
  echo "Evaluating dataset: ${DATASET}  |  Config: ${CFG}"
  echo "=========================================="

  python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}_${SUB}/${DATASET} \
    --eval-only \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

  echo "✅ Finished: ${DATASET}"
  echo ""
done

echo "🎯 All datasets evaluated with config: ${CFG}"
