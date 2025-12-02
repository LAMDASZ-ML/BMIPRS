DEVICE=cuda:1
CFG=$1

for SEED in 1 2 3
do
  # bash scripts/bmip/base2new_train_bmip.sh aid ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh aid ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip.sh optimal ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh optimal ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip.sh patternnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh patternnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip.sh whurs19 ${SEED} ${DEVICE} ${CFG}

  bash scripts/bmip/acc.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done