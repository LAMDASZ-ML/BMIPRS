DEVICE=cuda:0
CFG=BMIPMIX

# for SEED in 1 2 3
for SEED in 1
do
  bash scripts/bmipmix/base2new_train_bmipmix.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipmix/base2new_test_bmipmix.sh aid ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh optimal ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh optimal ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh patternnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh patternnet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh whurs19 ${SEED} ${DEVICE} ${CFG}


  # bash scripts/bmipmix/base2new_train_bmipmix.sh eurosat ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh eurosat ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh imagenet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh imagenet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh dtd ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  dtd ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh ucf101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  ucf101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh oxford_pets ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  oxford_pets ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh food101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  food101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh oxford_flowers ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  oxford_flowers ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh sun397 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  sun397 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh stanford_cars ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  stanford_cars ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_train_bmipmix.sh caltech101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmipmix/base2new_test_bmipmix.sh  caltech101 ${SEED} ${DEVICE} ${CFG}
done