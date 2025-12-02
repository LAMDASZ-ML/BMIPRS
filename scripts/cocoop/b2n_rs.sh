# for SEED in 1 2 3
for SEED in 3
do
    # bash scripts/cocoop/base2new_train.sh aid ${SEED}
    # bash scripts/cocoop/base2new_test.sh aid ${SEED}
    # bash scripts/cocoop/base2new_train.sh mlrsnet ${SEED} 
    # bash scripts/cocoop/base2new_test.sh mlrsnet ${SEED} 
    # bash scripts/cocoop/base2new_train.sh optimal ${SEED}
    # bash scripts/cocoop/base2new_test.sh optimal ${SEED}
    # bash scripts/cocoop/base2new_train.sh patternnet ${SEED}
    # bash scripts/cocoop/base2new_test.sh patternnet ${SEED}
    # bash scripts/cocoop/base2new_train.sh resisc45 ${SEED}
    # bash scripts/cocoop/base2new_test.sh  resisc45 ${SEED}
    # bash scripts/cocoop/base2new_train.sh rsicb128 ${SEED}
    # bash scripts/cocoop/base2new_test.sh  rsicb128 ${SEED}
    # bash scripts/cocoop/base2new_train.sh rsicb256 ${SEED}
    # bash scripts/cocoop/base2new_test.sh  rsicb256 ${SEED}
    # bash scripts/cocoop/base2new_train.sh whurs19 ${SEED}
    # bash scripts/cocoop/base2new_test.sh whurs19 ${SEED}

    # bash scripts/cocoop/acc.sh aid ${SEED}
    # bash scripts/cocoop/acc.sh mlrsnet ${SEED}
    # bash scripts/cocoop/acc.sh optimal ${SEED}
    # bash scripts/cocoop/acc.sh patternnet ${SEED}
    # bash scripts/cocoop/acc.sh resisc45 ${SEED}
    # bash scripts/cocoop/acc.sh rsicb128 ${SEED}
    # bash scripts/cocoop/acc.sh rsicb256 ${SEED}
    bash scripts/cocoop/acc.sh whurs19 ${SEED}
done