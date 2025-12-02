data=ucf101
DEVICE=cuda:0
CFG=BMIP_loss

for ((INIT_SEED=1; INIT_SEED<=40; INIT_SEED+=1))
do
  for SEED in 1 2 3
  do
    bash scripts/bmip/base2new_train_bmiploss1.sh ${data} ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
    bash scripts/bmip/base2new_test_bmiploss1.sh ${data} ${SEED} ${DEVICE} ${INIT_SEED} ${CFG}
  done
done