DEVICE=cuda:0
CFG=BMIPKL_KL2

for SEED in 1 2 3
do
  bash scripts/bmipkl/base2new_train_bmipkl.sh eurosat ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh eurosat ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh imagenet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh imagenet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh dtd ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  dtd ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh ucf101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  ucf101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh oxford_pets ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  oxford_pets ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh food101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  food101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh oxford_flowers ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  oxford_flowers ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh sun397 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  sun397 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh stanford_cars ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  stanford_cars ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh caltech101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh  caltech101 ${SEED} ${DEVICE} ${CFG}


  ###############
  bash scripts/bmipkl/base2new_train_bmipkl.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh aid ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh mlrsnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh optimal ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh patternnet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh resisc45 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh rsicb128 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh rsicb256 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_train_bmipkl.sh whurs19 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmipkl/base2new_test_bmipkl.sh whurs19 ${SEED} ${DEVICE} ${CFG}
done