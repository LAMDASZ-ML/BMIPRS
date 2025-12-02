
DEVICE=cuda:0
CFG=BMIPSRC_rs_adam_4
for SEED in 1 2 3
do
    # bash scripts/promptsrc/base2new_train_promptsrc.sh eurosat ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_test_promptsrc.sh eurosat ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_test_promptsrc.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh dtd ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  dtd ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh ucf101 ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  ucf101 ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh oxford_pets ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  oxford_pets ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh food101 ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  food101 ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh oxford_flowers ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  oxford_flowers ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh sun397 ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  sun397 ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh stanford_cars ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  stanford_cars ${SEED} ${DEVICE} ${CFG}
    # bash scripts/promptsrc/base2new_train_promptsrc.sh caltech101 ${SEED} ${DEVICE} ${CFG}
	# bash scripts/promptsrc/base2new_test_promptsrc.sh  caltech101 ${SEED} ${DEVICE} ${CFG}
    bash scripts/promptsrc/base2new_train_promptsrc.sh imagenet ${SEED} ${DEVICE} ${CFG}
	bash scripts/promptsrc/base2new_test_promptsrc.sh  imagenet ${SEED} ${DEVICE} ${CFG}
done