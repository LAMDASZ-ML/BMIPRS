DEVICE=cuda:0

CFG=BMIPKL_ucf
for SEED in 1 2 3
do
  bash scripts/bmipkl/base2new_train_bmipkl_ucf.sh ucf101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl_ucf.sh ucf101 ${SEED} ${DEVICE} ${CFG}
done

# CFG=BMIPKL_ucf1
# for SEED in 1 2 3
# do
#   bash scripts/bmipkl/base2new_train_bmipkl_ucf.sh ucf101 ${SEED} ${DEVICE} ${CFG}
#   bash scripts/bmipkl/base2new_test_bmipkl_ucf.sh ucf101 ${SEED} ${DEVICE} ${CFG}
# done