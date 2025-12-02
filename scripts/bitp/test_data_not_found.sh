CFG=$1
for SEED in 1 2 3
do
  bash scripts/bitp/base2new_train_bitp.sh dtd ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh  dtd ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh caltech101 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh caltech101 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh food101 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh food101 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh oxford_flowers ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh oxford_flowers ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh stanford_cars ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh  stanford_cars ${SEED} cuda:1 ${CFG}
done