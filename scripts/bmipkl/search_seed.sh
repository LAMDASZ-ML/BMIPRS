data=eurosat
DEVICE=cuda:0
CFG=BMIPKL

for ((INIT_SEED=92; INIT_SEED<=150; INIT_SEED+=1))
do
  for SEED in 1 2 3
  do
    bash scripts/bmipkl/base2new_train_bmipkl1.sh ${data} ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
    bash scripts/bmipkl/base2new_test_bmipkl1.sh ${data} ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  done
done