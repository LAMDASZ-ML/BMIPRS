DEVICE=cuda:0
CFG=BMIP_loss_00

for SEED in 1 2 3
do
  bash scripts/bmip/base2new_train_bmip_loss.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip_loss.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip_loss.sh whurs19 ${SEED} ${DEVICE} ${CFG}

  bash scripts/bmip/acc.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/acc.sh whurs19 ${SEED} ${DEVICE} ${CFG}

  #######################################
  # bash scripts/bmip/base2new_train_bmip_loss.sh eurosat ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh eurosat ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh imagenet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh imagenet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh dtd ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  dtd ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh ucf101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  ucf101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh oxford_pets ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  oxford_pets ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh food101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  food101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh oxford_flowers ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  oxford_flowers ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh sun397 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  sun397 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh stanford_cars ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  stanford_cars ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_train_bmip_loss.sh caltech101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bmip/base2new_test_bmip_loss.sh  caltech101 ${SEED} ${DEVICE} ${CFG}
done