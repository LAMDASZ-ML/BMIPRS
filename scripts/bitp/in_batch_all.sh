CFG=$1
for SEED in 1 2 3
do
  bash scripts/bitp/base2new_test_bitp_all_2.sh eurosat ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh dtd ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh caltech101 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh fgvc_aircraft ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh imagenet ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh ucf101 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh oxford_flowers ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh food101 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh stanford_cars ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh sun397 ${SEED} cuda:1 ${CFG}
  bash scripts/bitp/base2new_test_bitp_all_2.sh oxford_pets ${SEED} cuda:1 ${CFG}
done