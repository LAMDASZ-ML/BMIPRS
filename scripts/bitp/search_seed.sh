data=$1

for ((INIT_SEED=15; INIT_SEED<=35; INIT_SEED+=1))
do
  for SEED in 1 2 3
  do
    bash scripts/bitp/base2new_train_bitp1.sh ${data} ${SEED} cuda:0 ${INIT_SEED}
    bash scripts/bitp/base2new_test_bitp1.sh ${data} ${SEED} cuda:0 ${INIT_SEED}
  done
done