CFG=$1
DEVICE=cuda:1
for SEED in 1 2 3
do
  bash scripts/bitp/base2new_train_bitp.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done