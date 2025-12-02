data=eurosat
DEVICE=cuda:1
CFG=BMIP_ctx2_dp9_lr2

for ((INIT_SEED=41; INIT_SEED<=80; INIT_SEED+=1))
do
  for SEED in 1 2 3
  do
    bash scripts/bmip/base2new_train_bmip1.sh ${data} ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
    bash scripts/bmip/base2new_test_bmip1.sh ${data} ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  done
done