DEVICE=cuda:0
CFG=BMIPKL
# KL48 79.97
# KL3 79.54
for SEED in 1 2 3
do
  bash scripts/bmipkl/base2new_train_bmipkl.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done