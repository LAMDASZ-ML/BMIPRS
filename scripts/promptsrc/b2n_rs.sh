DEVICE=cuda:0
CFG=promptsrc_rs_sgd_4

for SEED in 1 2 3
do
  bash scripts/promptsrc/base2new_train_promptsrc.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc.sh whurs19 ${SEED} ${DEVICE} ${CFG}

  bash scripts/promptsrc/acc.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done