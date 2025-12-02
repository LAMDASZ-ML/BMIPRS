CFG=$1
DEVICE=$2
load=$3
for SEED in 1 
do
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenetv2 ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenet_sketch ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenet_a ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenet_r ${SEED} ${DEVICE} ${CFG} ${load}
done