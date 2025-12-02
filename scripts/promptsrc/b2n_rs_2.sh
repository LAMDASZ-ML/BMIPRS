DEVICE=cuda:1
CFG=BMIPSRC_rs_adam_4

for SEED in 1 2 3
do
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh aid ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh aid ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh optimal ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh optimal ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh patternnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh patternnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc_2.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc_2.sh whurs19 ${SEED} ${DEVICE} ${CFG}

  bash scripts/promptsrc/acc2.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc2.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc2.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc2.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc2.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc2.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc2.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/promptsrc/acc2.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done