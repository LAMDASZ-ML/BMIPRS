CFG=$1
Device=$2
load_epoch=$3
for SEED in 1
do
    bash scripts/independent-vlp/xd_test_ivlp.sh food101 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh caltech101 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh oxford_flowers ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh oxford_pets ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh stanford_cars ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh fgvc_aircraft ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh sun397 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh eurosat ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh ucf101 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh dtd ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/independent-vlp/xd_test_ivlp.sh imagenet ${SEED} ${Device} ${CFG} ${load_epoch}
done