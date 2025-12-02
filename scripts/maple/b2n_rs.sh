for SEED in 1 2 3
do
    # bash scripts/maple/base2new_train_maple.sh aid ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh aid ${SEED} cuda:0
    # bash scripts/maple/base2new_train_maple.sh mlrsnet ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh mlrsnet ${SEED} cuda:0
    # bash scripts/maple/base2new_train_maple.sh optimal ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh optimal ${SEED} cuda:0
    # bash scripts/maple/base2new_train_maple.sh patternnet ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh patternnet ${SEED} cuda:0
    # bash scripts/maple/base2new_train_maple.sh resisc45 ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh resisc45 ${SEED} cuda:0
    # bash scripts/maple/base2new_train_maple.sh rsicb128 ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh rsicb128 ${SEED} cuda:0
    # bash scripts/maple/base2new_train_maple.sh rsicb256 ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh rsicb256 ${SEED} cuda:0
    # bash scripts/maple/base2new_train_maple.sh whurs19 ${SEED} cuda:0
    # bash scripts/maple/base2new_test_maple.sh whurs19 ${SEED} cuda:0

    bash scripts/maple/acc.sh aid ${SEED} cuda:0
    bash scripts/maple/acc.sh mlrsnet ${SEED} cuda:0
    bash scripts/maple/acc.sh optimal ${SEED} cuda:0
    bash scripts/maple/acc.sh patternnet ${SEED} cuda:0
    bash scripts/maple/acc.sh resisc45 ${SEED} cuda:0
    bash scripts/maple/acc.sh rsicb128 ${SEED} cuda:0
    bash scripts/maple/acc.sh rsicb256 ${SEED} cuda:0
    bash scripts/maple/acc.sh whurs19 ${SEED} cuda:0
done