DEVICE=cuda:0
CFG=BMIP_rs_es

for SEED in 1 2 3
do
  bash scripts/bmip/base2new_train_bmip.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_noes.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done