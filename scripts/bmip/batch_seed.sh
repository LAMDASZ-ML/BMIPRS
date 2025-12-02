#!/bin/bash
# ===============================
# 用法:
#   bash batch_seed.sh <INIT_SEED>
# 例如:
#   bash batch_seed.sh 77
# ===============================

DEVICE=cuda:1
CFG=BMIP_ctx2_dp9_lr2

# 从命令行参数获取 INIT_SEED
INIT_SEED=$1

# 检查是否传入参数
if [ -z "$INIT_SEED" ]; then
  echo "❌ 请提供 INIT_SEED，例如: bash batch_seed.sh 77"
  exit 1
fi

echo "🚀 Running all datasets with INIT_SEED=$INIT_SEED"

for SEED in 1 2 3
do
  # bash scripts/bmip/base2new_train_bmip1.sh aid ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh aid ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip1.sh mlrsnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh mlrsnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip1.sh optimal ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh optimal ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip1.sh patternnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh patternnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip1.sh resisc45 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh resisc45 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip1.sh rsicb128 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh rsicb128 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip1.sh rsicb256 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh rsicb256 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip1.sh whurs19 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip1.sh whurs19 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}

  #######################################
  bash scripts/bmip/base2new_train_bmip1.sh eurosat ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh eurosat ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh fgvc_aircraft ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh fgvc_aircraft ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh imagenet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh imagenet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh dtd ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  dtd ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh ucf101 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  ucf101 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh oxford_pets ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  oxford_pets ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh food101 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  food101 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh oxford_flowers ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  oxford_flowers ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh sun397 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  sun397 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh stanford_cars ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  stanford_cars ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_train_bmip1.sh caltech101 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  bash scripts/bmip/base2new_test_bmip1.sh  caltech101 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
done