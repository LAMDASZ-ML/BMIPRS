#!/bin/bash
CFG=$1
DEVICE=$2

for SEED in 1 2 3
do
  # bash scripts/bmip/base2new_train_bmip_loss.sh aid ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh aid ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh mlrsnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh mlrsnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh optimal ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh optimal ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh patternnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh patternnet ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh resisc45 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh resisc45 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh rsicb128 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh rsicb128 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh rsicb256 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh rsicb256 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh whurs19 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh whurs19 ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}

  #######################################
  bash scripts/bmip/base2new_train_bmip_loss.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done